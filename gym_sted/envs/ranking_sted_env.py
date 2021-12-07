
import gym
import numpy
import random
import os

from gym import error, spaces, utils
from gym.utils import seeding
from matplotlib import pyplot
from collections import OrderedDict

import gym_sted
from gym_sted import rewards, defaults
from gym_sted.utils import SynapseGenerator, MicroscopeGenerator, get_foreground, BleachSampler, Normalizer
from gym_sted.rewards import objectives
from gym_sted.prefnet import load_demonstrations

from gym_sted.defaults import obj_dict, bounds_dict, scales_dict

class STEDMultiObjectivesEnv(gym.Env):
    """
    Creates a `STEDMultiObjectivesEnv`

    Action space
        The action space corresponds to the imaging parameters

    Observation space
        The observation space is a tuple, where
        1. The current confocal image, the previous confocal and STED images
        2. A vector of the selected actions and the obtained objectives in the episode

    """
    metadata = {'render.modes': ['human']}
    obj_names = ["Resolution", "Bleach", "SNR"]

    def __init__(self, bleach_sampling="constant", actions=["p_sted"],
                    max_episode_steps=10, scale_nanodomain_reward=1.,
                    normalize_observations=False):

        self.actions = actions
        self.action_space = spaces.Box(
            low=numpy.array([defaults.action_spaces[name]["low"] for name in self.actions]),
            high=numpy.array([defaults.action_spaces[name]["high"] for name in self.actions]),
            dtype=numpy.float32
        )

        self.observation_space = spaces.Tuple((
            spaces.Box(0, 2**16, shape=(64, 64, 3), dtype=numpy.uint16),
            spaces.Box(
                0, 1, shape=(len(self.obj_names) * max_episode_steps + len(self.actions) * max_episode_steps,),
                dtype=numpy.float32
            ) # Articulation, shape is given by objectives, actions at each steps
        ))

        self.state = None
        self.initial_count = None
        self.synapse = None
        self.current_step = 0
        self.bleach_sampling = bleach_sampling
        self.scale_nanodomain_reward = scale_nanodomain_reward
        self.episode_memory = {
            "actions" : [],
            "mo_objs" : [],
            "reward" : [],
        }

        self.datamap = None
        self.viewer = None

        # seed = self.seed(0)
        molecules = 5
        self.synapse_generator = SynapseGenerator(
            mode="mushroom", seed=None, n_nanodomains=(3, 15), n_molecs_in_domain=(molecules * 20, molecules * 35)
        )
        self.microscope_generator = MicroscopeGenerator()
        self.microscope = self.microscope_generator.generate_microscope()
        self.bleach_sampler = BleachSampler(mode=self.bleach_sampling)

        objs = OrderedDict({obj_name : obj_dict[obj_name] for obj_name in self.obj_names})
        bounds = OrderedDict({obj_name : bounds_dict[obj_name] for obj_name in self.obj_names})
        scales = OrderedDict({obj_name : scales_dict[obj_name] for obj_name in self.obj_names})
        self.mo_reward_calculator = rewards.MORewardCalculator(objs, bounds=bounds, scales=scales)
        self.nb_reward_calculator = rewards.NanodomainsRewardCalculator(
            {"NbNanodomains" : obj_dict["NbNanodomains"]},
            bounds={"NbNanodomains" : bounds_dict["NbNanodomains"]},
            scales={"NbNanodomains" : scales_dict["NbNanodomains"]}
        )

        # Creates an action and objective normalizer
        self.normalize_observations = normalize_observations
        self.action_normalizer = Normalizer(self.actions, defaults.action_spaces)
        self.obj_normalizer = Normalizer(self.obj_names, scales_dict)

    def step(self, action):
        """
        Method that should be implemented in the object that inherited

        :param action: A `numpy.ndarray` of the action
        """
        raise NotImplementedError

    def reset(self):
        """
        Resets the environment with a new datamap

        :returns : A `numpy.ndarray` of the molecules
        """
        # Updates the current bleach function
        self.microscope = self.microscope_generator.generate_microscope(
            phy_react=self.bleach_sampler.sample()
        )

        self.current_step = 0
        self.episode_memory = {
            "actions" : [],
            "mo_objs" : [],
            "reward" : [],
        }

        state = self._update_datamap()

        self.state = numpy.stack((state, numpy.zeros_like(state), numpy.zeros_like(state)), axis=-1)
        return (self.state, numpy.zeros((self.observation_space[1].shape[0],)))

    def render(self, info, mode='human'):
        """
        Renders the environment

        :param info: A `dict` of data
        :param mode: A `str` of the available mode
        """
        fig, axes = pyplot.subplots(1, 3, figsize=(10,3), sharey=True, sharex=True)

        axes[0].imshow(info["conf1"])
        axes[0].set_title(f"Datamap roi")

        axes[1].imshow(info["bleached"])
        axes[1].set_title(f"Bleached datamap")

        axes[2].imshow(info["sted_image"])
        axes[2].set_title(f"Acquired signal (photons)")

        pyplot.show(block=True)

    def seed(self, seed=None):
        """
        Seeds the environment

        :param seed: An `int` of the random seed
        """
        self.np_random, seed = seeding.np_random(seed)
        self.bleach_sampler.seed(seed)
        return [seed]

    def update_(self, **kwargs):
        """
        Utilitary method to update parameters in-place
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _update_datamap(self):
        """
        Generates a new `datamap` and acquires a confocal image. The new state of
        the environment is returned

        :returns : A `numpy.ndarray` of the state
        """
        self.synapse = self.synapse_generator(rotate=True)
        self.datamap = self.microscope_generator.generate_datamap(
            datamap = {
                "whole_datamap" : self.synapse.frame,
                "datamap_pixelsize" : self.microscope_generator.pixelsize
            }
        )

        # Acquire confocal image which sets the current state
        conf_params = self.microscope_generator.generate_params()
        state, _, _ = self.microscope.get_signal_and_bleach(
            self.datamap, self.datamap.pixelsize, **conf_params, bleach=False
        )
        return state

    def _acquire(self, action):
        """
        Acquires from the `datamap` using the provided parameters. A confocal image
        before and after the STED image are acquired to calculate the objectives

        :param action: A `numpy.ndarray` of the imaging parameters

        :returns : A `numpy.ndarray` of the acquired STED
                   A `numpy.ndarray` of the bleached `datamap`
                   A `numpy.ndarray` of the acquired conf1
                   A `numpy.ndarray` of the acquired conf2
                   A `numpy.ndarray` of the foreground in the STED image
                   A `numpy.ndarray` of the foreground in the conf1 image
        """
        # Generates imaging parameters
        sted_params = self.microscope_generator.generate_params(
            imaging = {
                name : action[self.actions.index(name)]
                    if name in self.actions else getattr(defaults, name.upper())
                    for name in ["pdt", "p_ex", "p_sted"]
            }
        )
        conf_params = self.microscope_generator.generate_params()

        # Acquire confocal image
        conf1, bleached, _ = self.microscope.get_signal_and_bleach(
            self.datamap, self.datamap.pixelsize, **conf_params, bleach=False
        )

        # Acquire STED image
        sted_image, bleached, _ = self.microscope.get_signal_and_bleach(
            self.datamap, self.datamap.pixelsize, **sted_params, bleach=True
        )

        # Acquire confocal image
        conf2, bleached, _ = self.microscope.get_signal_and_bleach(
            self.datamap, self.datamap.pixelsize, **conf_params, bleach=False
        )

        # foreground on confocal image
        fg_c = get_foreground(conf1)
        # foreground on sted image
        if numpy.any(sted_image):
            fg_s = get_foreground(sted_image)
        else:
            fg_s = numpy.ones_like(fg_c)
        # remove STED foreground points not in confocal foreground, if any
        fg_s *= fg_c

        return sted_image, bleached["base"][self.datamap.roi], conf1, conf2, fg_s, fg_c

    def close(self):
        return None

class ContextualSTEDMultiObjectivesEnv(STEDMultiObjectivesEnv):
    """
    Creates a `ContextualSTEDMultiObjectivesEnv`

    Action space
        The action space corresponds to the imaging parameters

    Observation space
        The observation space is a tuple, where
        1. The current confocal image, the previous confocal and STED images
        2. A vector of the selected actions and the obtained objectives in the episode
    """
    metadata = {'render.modes': ['human']}
    obj_names = ["Resolution", "Bleach", "SNR"]

    def __init__(self, bleach_sampling="constant", actions=["p_sted"],
                    max_episode_steps=10, scale_nanodomain_reward=1.,
                    normalize_observations=False):

        super(ContextualSTEDMultiObjectivesEnv, self).__init__(
            bleach_sampling = bleach_sampling,
            actions = actions,
            max_episode_steps = max_episode_steps,
            scale_nanodomain_reward = scale_nanodomain_reward,
            normalize_observations=normalize_observations
        )

    def step(self, action):
        # We manually clip the actions which are out of action space
        action = numpy.clip(action, self.action_space.low, self.action_space.high)

        # Acquire an image with the given parameters
        sted_image, bleached, conf1, conf2, fg_s, fg_c = self._acquire(action)
        mo_objs = self.mo_reward_calculator.evaluate(sted_image, conf1, conf2, fg_s, fg_c)
        f1_score = self.nb_reward_calculator.evaluate(sted_image, conf1, conf2, fg_s, fg_c, synapse=self.synapse)
        reward = f1_score * self.scale_nanodomain_reward

        # Updates memory
        done = self.current_step >= self.spec.max_episode_steps - 1
        self.current_step += 1
        self.episode_memory["mo_objs"].append(mo_objs)
        self.episode_memory["actions"].append(action)
        self.episode_memory["reward"].append(reward)

        state = self._update_datamap()
        self.state = numpy.stack((state, conf1, sted_image), axis=-1)

        info = {
            "action" : action,
            "bleached" : bleached,
            "sted_image" : sted_image,
            "conf1" : conf1,
            "conf2" : conf2,
            "fg_c" : fg_c,
            "fg_s" : fg_s,
            "mo_objs" : mo_objs,
            "reward" : reward,
            "f1-score" : f1_score,
            "nanodomains-coords" : self.synapse.nanodomains_coords
        }

        # Build the observation space
        obs = []
        for a, mo in zip(self.episode_memory["actions"], self.episode_memory["mo_objs"]):
            obs.extend(self.action_normalizer(a) if self.normalize_observations else a)
            obs.extend(self.obj_normalizer(mo) if self.normalize_observations else mo)
        obs = numpy.pad(numpy.array(obs), (0, self.observation_space[1].shape[0] - len(obs)))

        return (self.state, obs), reward, done, info

class ExpertDemonstrationF1ScoreSTEDMultiObjectivesEnv(STEDMultiObjectivesEnv):
    """
    Creates a `ExpertDemonstrationSTEDMultiObjectivesEnv`

    Action space
        The action space corresponds to the imaging parameters

    Observation space
        The observation space is a tuple, where
        1. The current confocal image, the previous confocal and STED images
        2. A vector of the selected actions and the obtained objectives in the episode
    """
    metadata = {'render.modes': ['human']}
    obj_names = ["Resolution", "Bleach", "SNR"]

    def __init__(self, bleach_sampling="constant", actions=["p_sted"],
                    max_episode_steps=10, scale_nanodomain_reward=1.,
                    normalize_observations=False):

        super(ExpertDemonstrationF1ScoreSTEDMultiObjectivesEnv, self).__init__(
            bleach_sampling = bleach_sampling,
            actions = actions,
            max_episode_steps = max_episode_steps,
            scale_nanodomain_reward = scale_nanodomain_reward,
            normalize_observations=normalize_observations
        )

        # Load expert demonstration
        self.demonstrations = load_demonstrations()

    def step(self, action):
        """
        Implements a single step in the environment

        :param action: A `numpy.ndarray` of the imaging parameters

        :returns : A `tuple` of the observation
                   A `float` of the received reward
                   A `bool` whether the episode is finished
                   A `dict` of information about the episode
        """
        # We manually clip the actions which are out of action space
        action = numpy.clip(action, self.action_space.low, self.action_space.high)

        # Acquire an image with the given parameters
        sted_image, bleached, conf1, conf2, fg_s, fg_c = self._acquire(action)
        mo_objs = self.mo_reward_calculator.evaluate(sted_image, conf1, conf2, fg_s, fg_c)
        f1_score = self.nb_reward_calculator.evaluate(sted_image, conf1, conf2, fg_s, fg_c, synapse=self.synapse)

        # Reward is proportionnal to the ranked position of the last image
        sorted_indices = numpy.argsort(self.demonstrations + [f1_score])
        index = numpy.argmax(sorted_indices).item()
        # Reward is given by the position in the sorting
        reward = (index + 1) / len(sorted_indices)

        # Updates memory
        done = self.current_step >= self.spec.max_episode_steps - 1
        self.current_step += 1
        self.episode_memory["mo_objs"].append(mo_objs)
        self.episode_memory["actions"].append(action)
        self.episode_memory["reward"].append(reward)

        state = self._update_datamap()
        self.state = numpy.stack((state, conf1, sted_image), axis=-1)

        info = {
            "action" : action,
            "bleached" : bleached,
            "sted_image" : sted_image,
            "conf1" : conf1,
            "conf2" : conf2,
            "fg_c" : fg_c,
            "fg_s" : fg_s,
            "mo_objs" : mo_objs,
            "reward" : reward,
            "f1-score" : f1_score,
            "nanodomains-coords" : self.synapse.nanodomains_coords
        }

        # Build the observation space
        obs = []
        for a, mo in zip(self.episode_memory["actions"], self.episode_memory["mo_objs"]):
            obs.extend(self.action_normalizer(a) if self.normalize_observations else a)
            obs.extend(self.obj_normalizer(mo) if self.normalize_observations else mo)
        obs = numpy.pad(numpy.array(obs), (0, self.observation_space[1].shape[0] - len(obs)))

        return (self.state, obs), reward, done, info

class HumanSTEDMultiObjectivesEnv(STEDMultiObjectivesEnv):
    """
    Creates a `HumanSTEDMultiObjectivesEnv`

    Action space
        The action space corresponds to the imaging parameters

    Observation space
        The observation space is a tuple, where
        1. The current confocal image, the previous confocal and STED images
        2. A vector of the selected actions and the obtained objectives in the episode
    """
    metadata = {'render.modes': ['human']}
    obj_names = ["Resolution", "Bleach", "SNR"]

    def __init__(self, bleach_sampling="constant", actions=["p_sted"],
                    max_episode_steps=10, scale_nanodomain_reward=1.,
                    normalize_observations=False):

        super(HumanSTEDMultiObjectivesEnv, self).__init__(
            bleach_sampling = bleach_sampling,
            actions = actions,
            max_episode_steps = max_episode_steps,
            scale_nanodomain_reward = scale_nanodomain_reward,
            normalize_observations = normalize_observations
        )

    def step(self, action):
        """
        Implements a single step in the environment

        :param action: A `numpy.ndarray` of the imaging parameters

        :returns : A `tuple` of the observation
                   A `float` of the received reward
                   A `bool` whether the episode is finished
                   A `dict` of information about the episode
        """
        # We manually clip the actions which are out of action space
        action = numpy.clip(action, self.action_space.low, self.action_space.high)

        # On the last step of the environment we enforce the final decision
        final_action = self.current_step >= self.spec.max_episode_steps - 1

        # Acquire an image with the given parameters
        sted_image, bleached, conf1, conf2, fg_s, fg_c = self._acquire(action)
        mo_objs = self.mo_reward_calculator.evaluate(sted_image, conf1, conf2, fg_s, fg_c)
        f1_score = self.nb_reward_calculator.evaluate(sted_image, conf1, conf2, fg_s, fg_c, synapse=self.synapse)

        reward = f1_score
        done = False

        if final_action:
            reward += f1_score * self.scale_nanodomain_reward
            done = True

        # Updates memory
        self.current_step += 1
        self.episode_memory["mo_objs"].append(mo_objs)
        self.episode_memory["actions"].append(action)
        self.episode_memory["reward"].append(reward)

        state = self._update_datamap()
        self.state = numpy.stack((state, conf1, sted_image), axis=-1)

        info = {
            "action" : action,
            "bleached" : bleached,
            "sted_image" : sted_image,
            "conf1" : conf1,
            "conf2" : conf2,
            "fg_c" : fg_c,
            "fg_s" : fg_s,
            "mo_objs" : mo_objs,
            "reward" : reward,
            "f1-score" : f1_score,
            "nanodomains-coords" : self.synapse.nanodomains_coords
        }

        # Build the observation space
        obs = []
        for a, mo in zip(self.episode_memory["actions"], self.episode_memory["mo_objs"]):
            obs.extend(self.action_normalizer(numpy.array(a)) if self.normalize_observations else a)
            obs.extend(self.obj_normalizer(numpy.array(mo)) if self.normalize_observations else mo)
        obs = numpy.pad(numpy.array(obs), (0, self.observation_space[1].shape[0] - len(obs)))

        return (self.state, obs), reward, done, info
