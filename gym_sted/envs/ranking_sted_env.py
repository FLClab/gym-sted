
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
from gym_sted.prefnet import PreferenceArticulator

obj_dict = {
    "SNR" : objectives.Signal_Ratio(75),
    "Bleach" : objectives.Bleach(),
    "Resolution" : objectives.Resolution(pixelsize=20e-9),
    "Squirrel" : objectives.Squirrel(),
    "NbNanodomains" : objectives.NumberNanodomains()
}
bounds_dict = {
    "SNR" : {"low" : 0.20, "high" : numpy.inf},
    "Bleach" : {"low" : -numpy.inf, "high" : 0.5},
    "Resolution" : {"low" : 0, "high" : 100},
    "NbNanodomains" : {"low" : 0, "high" : numpy.inf}
}
scales_dict = {
    "SNR" : {"low" : 0, "high" : 1},
    "Bleach" : {"low" : 0, "high" : 1},
    "Resolution" : {"low" : 40, "high" : 180},
    "NbNanodomains" : {"low" : 0, "high" : 1}
}

class rankSTEDSingleObjectiveEnv(gym.Env):
    """
    Creates a `rankSTEDSingleObjectiveEnv`

    Action space
        The action space here is a tuple, where
        1. The first element of the tuple represents the imaging parameter selection
        2. The second element of the tuple represents the action to choose from
            {0: Acquire image, 1: Ask for expert knowledge, 2: Final parameters}

    """
    metadata = {'render.modes': ['human']}
    obj_names = ["Resolution", "Bleach", "SNR"]

    def __init__(self, reward_calculator="SumRewardCalculator", actions=["p_sted"],
                    max_num_requests=1, max_episode_steps=10, select_final=True):

        self.synapse_generator = SynapseGenerator(mode="mushroom", seed=None)
        self.microscope_generator = MicroscopeGenerator()
        self.microscope = self.microscope_generator.generate_microscope()

        self.actions = actions
        if self.select_final:
            self.action_space = spaces.Box(
                low=numpy.array([defaults.action_spaces[name]["low"] for name in self.actions] + [0]),
                high=numpy.array([defaults.action_spaces[name]["high"] for name in self.actions] + [2 + 1]),
                dtype=numpy.float32
            )
        else:
            self.action_space = spaces.Box(
                low=numpy.array([defaults.action_spaces[name]["low"] for name in self.actions] + [0]),
                high=numpy.array([defaults.action_spaces[name]["high"] for name in self.actions] + [1 + 1]),
                dtype=numpy.float32
            )

        self.observation_space = spaces.Tuple((
            spaces.Box(0, 2**16, shape=(64, 64, 1), dtype=numpy.uint16),
            spaces.Box(0, 1, shape=(max_episode_steps,), dtype=numpy.float32)
        ))

        self.state = None
        self.synapse = None
        self.initial_count = None
        self.current_step = 0
        self.max_num_requests = max_num_requests
        self.num_request_left = max_num_requests
        self.current_articulation = -1
        self.cummulated_rewards = {
            "rewards" : [],
            "reward" : []
        }

        objs = OrderedDict({obj_name : obj_dict[obj_name] for obj_name in self.obj_names})
        bounds = OrderedDict({obj_name : bounds_dict[obj_name] for obj_name in self.obj_names})
        scales = OrderedDict({obj_name : scales_dict[obj_name] for obj_name in self.obj_names})
        self.reward_calculator = getattr(rewards, reward_calculator)(objs, bounds=bounds, scales=scales)
        self._reward_calculator = rewards.MORewardCalculator(objs, bounds=bounds, scales=scales)

        # Loads preference articulation model
        self.preference_articulation = PreferenceArticulator()

        self.datamap = None
        self.viewer = None

        self.seed()

    def step(self, action):

        # Action is an array of size self.actions and main_action
        # main action should be in the [0, 1, 2]
        # We manually clip the actions which are out of action space
        action = numpy.clip(action, self.action_space.low, self.action_space.high)
        imaging_action, main_action = action[:len(self.actions)], min(int(action[-1]), 2)
        # On the last step of the environment we enforce the final decision
        if self.current_step >= self.spec.max_episode_steps - 1:
            main_action = 2

        if (main_action == 0) or (main_action == 1 and self.num_request_left <= 0):
            # Acquire an image with the given parameters
            sted_image, bleached, conf1, conf2, fg_s, fg_c = self._acquire(imaging_action)

            done = False
            reward = self.reward_calculator.evaluate(sted_image, conf1, conf2, fg_s, fg_c)
            rewards = self._reward_calculator.evaluate(sted_image, conf1, conf2, fg_s, fg_c)

        elif main_action == 1:
            # Asking for preference
            sted_image = numpy.zeros(self.observation_space[0].shape[:-1])
            bleached = numpy.zeros(self.observation_space[0].shape[:-1])
            conf1 = numpy.zeros(self.observation_space[0].shape[:-1])
            conf2 = numpy.zeros(self.observation_space[0].shape[:-1])
            fg_s = numpy.zeros(self.observation_space[0].shape[:-1])
            fg_c = numpy.zeros(self.observation_space[0].shape[:-1])

            done = False
            reward = 1
            rewards = [scales_dict[obj_name]["high"] if obj_name != "SNR" else scales_dict[obj_name]["low"] for obj_name in self.obj_names]
            self.num_request_left -= 1

            if len(self.cummulated_rewards["rewards"]) > 0:
                self.current_articulation, _ = self.preference_articulation.articulate(self.cummulated_rewards["rewards"])

        elif main_action == 2:
            # Acquire final image
            sted_image, bleached, conf1, conf2, fg_s, fg_c = self._acquire(imaging_action)
            reward = self.reward_calculator.evaluate(sted_image, conf1, conf2, fg_s, fg_c)
            rewards = self._reward_calculator.evaluate(sted_image, conf1, conf2, fg_s, fg_c)

            # Reward is proportionnal to the ranked position of the last image
            articulation, sorted_indices = self.preference_articulation.articulate(
                self.cummulated_rewards["rewards"] + [rewards]
            )
            index = numpy.argmax(sorted_indices).item()
            # Reward is given by the position in the sorting
            reward += (index + 1) / len(sorted_indices)

            done = True

        else:
            raise NotImplementedError("Only 3 actions supported...")

        self.current_step += 1
        self.cummulated_rewards["rewards"].append(rewards)
        self.cummulated_rewards["reward"].append(reward)

        state = self._update_datamap()
        self.state = state[..., numpy.newaxis]

        info = {
            "action" : action,
            "bleached" : bleached,
            "sted_image" : sted_image,
            "conf1" : conf1,
            "conf2" : conf2,
            "fg_c" : fg_c,
            "fg_s" : fg_s,
            "rewards" : rewards,
            "articulation" : self.current_articulation
        }
        if self.current_articulation == -1:
            articulation = numpy.zeros((self.spec.max_episode_steps, ))
        else:
            articulation = numpy.eye(self.spec.max_episode_steps)[self.current_articulation]
        return (self.state, articulation), reward, done, info

    def reset(self):
        """
        Resets the environment with a new datamap

        :returns : A `numpy.ndarray` of the molecules
        """
        self.num_request_left = self.max_num_requests
        self.current_step = 0
        self.current_articulation = -1
        self.cummulated_rewards = {
            "rewards" : [],
            "reward" : []
        }

        state = self._update_datamap()

        self.state = state[..., numpy.newaxis]
        return (self.state, numpy.zeros((self.spec.max_episode_steps,)))

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
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def update_(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _update_datamap(self):
        self.synapse = self.synapse_generator()
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

class STEDMultiObjectivesEnv(gym.Env):
    """
    Creates a `STEDMultiObjectivesEnv`

    Action space
        The action space corresponds to the imaging parameters

    Observation space
        The observation space is a tuple, where
        1. The current confocal image
        2. A vector containing the current articulation, the selected actions, the obtained objectives

    """
    metadata = {'render.modes': ['human']}
    obj_names = ["Resolution", "Bleach", "SNR"]

    def __init__(self, bleach_sampling="constant", actions=["p_sted"],
                    max_episode_steps=10, scale_nanodomain_reward=1.):

        self.actions = actions
        self.action_space = spaces.Box(
            low=numpy.array([defaults.action_spaces[name]["low"] for name in self.actions]),
            high=numpy.array([defaults.action_spaces[name]["high"] for name in self.actions]),
            dtype=numpy.float32
        )

        self.observation_space = spaces.Tuple((
            spaces.Box(0, 2**16, shape=(64, 64, 1), dtype=numpy.uint16),
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

        self.seed()

        self.synapse_generator = SynapseGenerator(mode="mushroom", seed=None)
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

        # Loads preference articulation model
        self.preference_articulation = PreferenceArticulator()

        # Creates an action and objective normalizer
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

        self.state = state[..., numpy.newaxis]
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
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def update_(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _update_datamap(self):
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

class rankSTEDMultiObjectivesEnv(STEDMultiObjectivesEnv):
    """
    Creates a `rankSTEDMultiObjectivesEnv`

    Action space
        The action space corresponds to the imaging parameters

    Observation space
        The observation space is a tuple, where
        1. The current confocal image
        2. A vector containing the current articulation, the selected actions, the obtained objectives

    """
    metadata = {'render.modes': ['human']}
    obj_names = ["Resolution", "Bleach", "SNR"]

    def __init__(self, bleach_sampling="constant", actions=["p_sted"],
                    max_episode_steps=10, scale_nanodomain_reward=1.):

        super(rankSTEDMultiObjectivesEnv, self).__init__(
            bleach_sampling = bleach_sampling,
            actions = actions,
            max_episode_steps = max_episode_steps,
            scale_nanodomain_reward = scale_nanodomain_reward
        )

    def step(self, action):
        # Action is an array of size self.actions and main_action
        # main action should be in the [0, 1, 2]
        # We manually clip the actions which are out of action space
        action = numpy.clip(action, self.action_space.low, self.action_space.high)

        # On the last step of the environment we enforce the final decision
        final_action = self.current_step >= self.spec.max_episode_steps - 1

        # Acquire an image with the given parameters
        sted_image, bleached, conf1, conf2, fg_s, fg_c = self._acquire(action)
        mo_objs = self.mo_reward_calculator.evaluate(sted_image, conf1, conf2, fg_s, fg_c)
        f1_score = self.nb_reward_calculator.evaluate(sted_image, conf1, conf2, fg_s, fg_c, synapse=self.synapse)
        if final_action:
            reward = f1_score
            reward = reward * self.scale_nanodomain_reward

            done = True
        else:
            # Reward is proportionnal to the ranked position of the last image
            articulation, sorted_indices = self.preference_articulation.articulate(
                self.episode_memory["mo_objs"] + [mo_objs]
            )
            index = numpy.argmax(sorted_indices).item()
            # Reward is given by the position in the sorting
            reward = (index + 1) / len(sorted_indices)

            done = False

        # Updates memory
        self.current_step += 1
        self.episode_memory["mo_objs"].append(mo_objs)
        self.episode_memory["actions"].append(action)
        self.episode_memory["reward"].append(reward)

        state = self._update_datamap()
        self.state = state[..., numpy.newaxis]

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
            obs.extend(a)
            obs.extend(mo)
        obs = numpy.pad(numpy.array(obs), (0, self.observation_space[1].shape[0] - len(obs)))

        return (self.state, obs), reward, done, info

class rankSTEDRecurrentMultiObjectivesEnv(STEDMultiObjectivesEnv):
    """
    Creates a `rankSTEDMultiObjectivesEnv`

    Action space
        The action space corresponds to the imaging parameters

    Observation space
        The observation space is a tuple, where
        1. The current confocal image
        2. A vector containing the current articulation, the selected actions, the obtained objectives

    """
    metadata = {'render.modes': ['human']}
    obj_names = ["Resolution", "Bleach", "SNR"]

    def __init__(self, bleach_sampling="constant", actions=["p_sted"],
                    max_episode_steps=10, scale_nanodomain_reward=1.):

        super(rankSTEDRecurrentMultiObjectivesEnv, self).__init__(
            bleach_sampling = bleach_sampling,
            actions = actions,
            max_episode_steps = max_episode_steps,
            scale_nanodomain_reward = scale_nanodomain_reward
        )

        # We redefine the observation space in case of recurrent model
        self.observation_space = spaces.Tuple((
            spaces.Box(0, 2**16, shape=(64, 64, 1), dtype=numpy.uint16),
            spaces.Box(
                0, 1, shape=(len(self.obj_names) + len(self.actions),),
                dtype=numpy.float32
            ) # Articulation, shape is given by objectives, actions at each steps
        ))

    def step(self, action):

        # Action is an array of size self.actions and main_action
        # main action should be in the [0, 1, 2]
        # We manually clip the actions which are out of action space
        action = numpy.clip(action, self.action_space.low, self.action_space.high)

        # On the last step of the environment we enforce the final decision
        final_action = self.current_step >= self.spec.max_episode_steps - 1

        # Acquire an image with the given parameters
        sted_image, bleached, conf1, conf2, fg_s, fg_c = self._acquire(action)
        mo_objs = self.mo_reward_calculator.evaluate(sted_image, conf1, conf2, fg_s, fg_c)
        f1_score = self.nb_reward_calculator.evaluate(sted_image, conf1, conf2, fg_s, fg_c, synapse=self.synapse)
        if final_action:
            reward = f1_score
            reward = reward * self.scale_nanodomain_reward

            done = True
        else:
            # Reward is proportionnal to the ranked position of the last image
            articulation, sorted_indices = self.preference_articulation.articulate(
                self.episode_memory["mo_objs"] + [mo_objs]
            )
            index = numpy.argmax(sorted_indices).item()
            # Reward is given by the position in the sorting
            reward = (index + 1) / len(sorted_indices)

            done = False

        # Updates memory
        self.current_step += 1
        self.episode_memory["mo_objs"].append(mo_objs)
        self.episode_memory["actions"].append(action)
        self.episode_memory["reward"].append(reward)

        state = self._update_datamap()
        self.state = state[..., numpy.newaxis]

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
        obs = numpy.concatenate((self.episode_memory["actions"][-1], self.episode_memory["mo_objs"][-1]), axis=0)

        return (self.state, obs), reward, done, info

class rankSTEDMultiObjectivesWithDelayedRewardEnv(STEDMultiObjectivesEnv):
    """
    Creates a `rankSTEDMultiObjectivesWithDelayedRewardEnv`

    Action space
        The action space corresponds to the imaging parameters

    Observation space
        The observation space is a tuple, where
        1. The current confocal image
        2. A vector containing the current articulation, the selected actions, the obtained objectives

    """
    metadata = {'render.modes': ['human']}
    obj_names = ["Resolution", "Bleach", "SNR"]

    def __init__(self, bleach_sampling="constant", actions=["p_sted"],
                    max_episode_steps=10, scale_nanodomain_reward=1.):

        super(rankSTEDMultiObjectivesWithDelayedRewardEnv, self).__init__(
            bleach_sampling = bleach_sampling,
            actions = actions,
            max_episode_steps = max_episode_steps,
            scale_nanodomain_reward = scale_nanodomain_reward
        )

    def step(self, action):
        # Action is an array of size self.actions and main_action
        # main action should be in the [0, 1, 2]
        # We manually clip the actions which are out of action space
        action = numpy.clip(action, self.action_space.low, self.action_space.high)

        # On the last step of the environment we enforce the final decision
        final_action = self.current_step >= self.spec.max_episode_steps - 1

        # Acquire an image with the given parameters
        sted_image, bleached, conf1, conf2, fg_s, fg_c = self._acquire(action)
        mo_objs = self.mo_reward_calculator.evaluate(sted_image, conf1, conf2, fg_s, fg_c)
        f1_score = self.nb_reward_calculator.evaluate(sted_image, conf1, conf2, fg_s, fg_c, synapse=self.synapse)
        if final_action:

            # Reward is proportionnal to the ranked position of the last image
            _, sorted_indices = self.preference_articulation.articulate(
                self.episode_memory["mo_objs"] + [mo_objs]
            )
            index = numpy.argmax(sorted_indices).item()
            reward = self.nb_reward_calculator.evaluate(sted_image, conf1, conf2, fg_s, fg_c, synapse=self.synapse)
            reward = reward * self.scale_nanodomain_reward

            weights = numpy.linspace(0, 1, num=len(sorted_indices))
            weights = 1 - weights[index] + weights
            weights = weights[[numpy.argwhere(sorted_indices == idx).item() for idx in range(len(weights))]]
            reward = reward * weights

            done = True
        else:
            reward = 0
            done = False

        # Updates memory
        self.current_step += 1
        self.episode_memory["mo_objs"].append(mo_objs)
        self.episode_memory["actions"].append(action)
        self.episode_memory["reward"].append(reward)

        state = self._update_datamap()
        self.state = state[..., numpy.newaxis]

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
            obs.extend(a)
            obs.extend(mo)
        obs = numpy.pad(numpy.array(obs), (0, self.observation_space[1].shape[0] - len(obs)))

        return (self.state, obs), reward, done, info

class ContextualSTEDMultiObjectivesEnv(STEDMultiObjectivesEnv):
    """
    Creates a `ContextualSTEDMultiObjectivesEnv`

    Action space
        The action space corresponds to the imaging parameters

    Observation space
        The observation space is a tuple, where
        1. The current confocal image
        2. A vector containing the current articulation, the selected actions, the obtained objectives

    """
    metadata = {'render.modes': ['human']}
    obj_names = ["Resolution", "Bleach", "SNR"]

    def __init__(self, bleach_sampling="constant", actions=["p_sted"],
                    max_episode_steps=10, scale_nanodomain_reward=1.):

        super(ContextualSTEDMultiObjectivesEnv, self).__init__(
            bleach_sampling = bleach_sampling,
            actions = actions,
            max_episode_steps = max_episode_steps,
            scale_nanodomain_reward = scale_nanodomain_reward
        )

    def step(self, action):

        # Action is an array of size self.actions and main_action
        # main action should be in the [0, 1, 2]
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
        self.state = state[..., numpy.newaxis]

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
            obs.extend(a)
            obs.extend(mo)
        obs = numpy.pad(numpy.array(obs), (0, self.observation_space[1].shape[0] - len(obs)))

        return (self.state, obs), reward, done, info

class ContextualRecurrentSTEDMultiObjectivesEnv(STEDMultiObjectivesEnv):
    """
    Creates a `ContextualRecurrentSTEDMultiObjectivesEnv`

    Action space
        The action space corresponds to the imaging parameters

    Observation space
        The observation space is a tuple, where
        1. The current confocal image
        2. A vector containing the current articulation, the selected actions, the obtained objectives

    """
    metadata = {'render.modes': ['human']}
    obj_names = ["Resolution", "Bleach", "SNR"]

    def __init__(self, bleach_sampling="constant", actions=["p_sted"],
                    max_episode_steps=10, scale_nanodomain_reward=1.):

        super(ContextualRecurrentSTEDMultiObjectivesEnv, self).__init__(
            bleach_sampling = bleach_sampling,
            actions = actions,
            max_episode_steps = max_episode_steps,
            scale_nanodomain_reward = scale_nanodomain_reward
        )

        # We redefine the observation space in case of recurrent model
        self.observation_space = spaces.Tuple((
            spaces.Box(0, 2**16, shape=(64, 64, 1), dtype=numpy.uint16),
            spaces.Box(
                0, 1, shape=(len(self.obj_names) + len(self.actions),),
                dtype=numpy.float32
            ) # Articulation, shape is given by objectives, actions at each steps
        ))

    def step(self, action):

        # Action is an array of size self.actions and main_action
        # main action should be in the [0, 1, 2]
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
        self.state = state[..., numpy.newaxis]

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
        obs = numpy.concatenate((
            self.action_normalizer(self.episode_memory["actions"][-1]),
            self.obj_normalizer(self.episode_memory["mo_objs"][-1])
        ), axis=0)

        return (self.state, obs), reward, done, info

class ContextualRankingSTEDMultiObjectivesEnv(STEDMultiObjectivesEnv):
    """
    Creates a `ContextualRankingMultiObjectivesEnv`

    Action space
        The action space corresponds to the imaging parameters

    Observation space
        The observation space is a tuple, where
        1. The current confocal image
        2. A vector containing the current articulation, the selected actions, the obtained objectives
    """
    metadata = {'render.modes': ['human']}
    obj_names = ["Resolution", "Bleach", "SNR"]

    def __init__(self, bleach_sampling="constant", actions=["p_sted"],
                    max_episode_steps=10, scale_nanodomain_reward=1.):

        super(ContextualRankingSTEDMultiObjectivesEnv, self).__init__(
            bleach_sampling = bleach_sampling,
            actions = actions,
            max_episode_steps = max_episode_steps,
            scale_nanodomain_reward = scale_nanodomain_reward
        )

    def step(self, action):

        # Action is an array of size self.actions and main_action
        # main action should be in the [0, 1, 2]
        # We manually clip the actions which are out of action space
        action = numpy.clip(action, self.action_space.low, self.action_space.high)

        # Acquire an image with the given parameters
        sted_image, bleached, conf1, conf2, fg_s, fg_c = self._acquire(action)
        mo_objs = self.mo_reward_calculator.evaluate(sted_image, conf1, conf2, fg_s, fg_c)
        f1_score = self.nb_reward_calculator.evaluate(sted_image, conf1, conf2, fg_s, fg_c, synapse=self.synapse)

        # Reward is proportionnal to the ranked position of the last image
        articulation, sorted_indices = self.preference_articulation.articulate(
            self.episode_memory["mo_objs"] + [mo_objs]
        )
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
        self.state = state[..., numpy.newaxis]

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
            obs.extend(a)
            obs.extend(mo)
        obs = numpy.pad(numpy.array(obs), (0, self.observation_space[1].shape[0] - len(obs)))

        return (self.state, obs), reward, done, info

class rankSTEDMultiObjectivesWithArticulationEnv(gym.Env):
    """
    Creates a `rankSTEDMultiObjectivesEnv`

    Action space
        The action space is a tuple, where
        1. The imaging parameter selection
        2. The action to choose from
            {0: Acquire image, 1: Ask for expert knowledge, 2: Final parameters}

    Observation space
        The observation space is a tuple, where
        1. The current confocal image
        2. A vector containing the current articulation, the selected actions, the obtained objectives

    """
    metadata = {'render.modes': ['human']}
    obj_names = ["Resolution", "Bleach", "SNR"]

    def __init__(self, bleach_sampling="constant", actions=["p_sted"],
                    max_num_requests=1, max_episode_steps=10, select_final=False,
                    scale_rank_reward=False, scale_nanodomain_reward=1.):

        self.actions = actions
        self.select_final = select_final
        self.scale_nanodomain_reward = scale_nanodomain_reward

        if self.select_final:
            self.action_space = spaces.Box(
                low=numpy.array([defaults.action_spaces[name]["low"] for name in self.actions] + [0]),
                high=numpy.array([defaults.action_spaces[name]["high"] for name in self.actions] + [2 + 1]),
                dtype=numpy.float32
            )
        else:
            self.action_space = spaces.Box(
                low=numpy.array([defaults.action_spaces[name]["low"] for name in self.actions] + [0]),
                high=numpy.array([defaults.action_spaces[name]["high"] for name in self.actions] + [1 + 1]),
                dtype=numpy.float32
            )

        self.observation_space = spaces.Tuple((
            spaces.Box(0, 2**16, shape=(64, 64, 1), dtype=numpy.uint16),
            spaces.Box(
                0, 1, shape=(max_episode_steps + len(self.obj_names) * max_episode_steps + len(self.actions) * max_episode_steps,),
                dtype=numpy.float32
            ) # Articulation, shape is given by articulation vector, objectives, actions
        ))

        self.state = None
        self.initial_count = None
        self.synapse = None
        self.current_step = 0
        self.max_num_requests = max_num_requests
        self.num_request_left = max_num_requests
        self.bleach_sampling = bleach_sampling
        self.scale_rank_reward = scale_rank_reward
        self.current_articulation = -1
        self.episode_memory = {
            "actions" : [],
            "mo_objs" : [],
            "reward" : [],
        }

        self.datamap = None
        self.viewer = None

        self.seed()

        self.synapse_generator = SynapseGenerator(mode="mushroom", seed=None)
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

        # Loads preference articulation model
        self.preference_articulation = PreferenceArticulator()

    def step(self, action):

        # Action is an array of size self.actions and main_action
        # main action should be in the [0, 1, 2]
        # We manually clip the actions which are out of action space
        action = numpy.clip(action, self.action_space.low, self.action_space.high)
        if self.select_final:
            imaging_action, main_action = action[:len(self.actions)], min(int(action[-1]), 2)
        else:
            imaging_action, main_action = action[:len(self.actions)], min(int(action[-1]), 1)

        # On the last step of the environment we enforce the final decision
        if self.current_step >= self.spec.max_episode_steps - 1:
            main_action = 2

        if (main_action == 0) or (main_action == 1 and self.num_request_left <= 0):
            # Acquire an image with the given parameters
            sted_image, bleached, conf1, conf2, fg_s, fg_c = self._acquire(imaging_action)
            mo_objs = self.mo_reward_calculator.evaluate(sted_image, conf1, conf2, fg_s, fg_c)

            # Reward is proportionnal to the ranked position of the last image
            articulation, sorted_indices = self.preference_articulation.articulate(
                self.episode_memory["mo_objs"] + [mo_objs]
            )
            index = numpy.argmax(sorted_indices).item()
            # Reward is given by the position in the sorting
            reward = (index + 1) / len(sorted_indices)

            done = False

        elif main_action == 1:
            # Asking for preference
            sted_image = numpy.zeros(self.observation_space[0].shape[:-1])
            bleached = numpy.zeros(self.observation_space[0].shape[:-1])
            conf1 = numpy.zeros(self.observation_space[0].shape[:-1])
            conf2 = numpy.zeros(self.observation_space[0].shape[:-1])
            fg_s = numpy.zeros(self.observation_space[0].shape[:-1])
            fg_c = numpy.zeros(self.observation_space[0].shape[:-1])

            reward = 1
            mo_objs = [scales_dict[obj_name]["high"] if obj_name != "SNR" else scales_dict[obj_name]["low"] for obj_name in self.obj_names]
            self.num_request_left -= 1

            if len(self.episode_memory["mo_objs"]) > 0:
                self.current_articulation, _ = self.preference_articulation.articulate(self.episode_memory["mo_objs"])

            done = False

        elif main_action == 2:
            # Acquire final image
            sted_image, bleached, conf1, conf2, fg_s, fg_c = self._acquire(imaging_action)
            mo_objs = self.mo_reward_calculator.evaluate(sted_image, conf1, conf2, fg_s, fg_c)

            reward = self.nb_reward_calculator.evaluate(sted_image, conf1, conf2, fg_s, fg_c, synapse=self.synapse)
            reward = reward * self.scale_nanodomain_reward

            done = True

        else:
            raise NotImplementedError("Only 3 actions supported...")

        # Updates memory
        self.current_step += 1
        self.episode_memory["mo_objs"].append(mo_objs)
        self.episode_memory["actions"].append(imaging_action)
        self.episode_memory["reward"].append(reward)

        state = self._update_datamap()
        self.state = state[..., numpy.newaxis]

        info = {
            "action" : action,
            "bleached" : bleached,
            "sted_image" : sted_image,
            "conf1" : conf1,
            "conf2" : conf2,
            "fg_c" : fg_c,
            "fg_s" : fg_s,
            "mo_objs" : mo_objs,
            "articulation" : self.current_articulation,
            "reward" : reward
        }
        if self.current_articulation == -1:
            articulation = numpy.zeros((self.spec.max_episode_steps, ))
        else:
            articulation = numpy.eye(self.spec.max_episode_steps)[self.current_articulation]

        # Build the observation space
        obs = []
        obs.extend(articulation)
        for a, mo in zip(self.episode_memory["actions"], self.episode_memory["mo_objs"]):
            obs.extend(a)
            obs.extend(mo)
        obs = numpy.pad(numpy.array(obs), (0, self.observation_space[1].shape[0] - len(obs)))

        return (self.state, obs), reward, done, info

    def reset(self):
        """
        Resets the environment with a new datamap

        :returns : A `numpy.ndarray` of the molecules
        """
        # Updates the current bleach function
        self.microscope = self.microscope_generator.generate_microscope(
            phy_react=self.bleach_sampler.sample()
        )

        self.num_request_left = self.max_num_requests
        self.current_step = 0
        self.current_articulation = -1
        self.episode_memory = {
            "actions" : [],
            "mo_objs" : [],
            "reward" : [],
        }

        state = self._update_datamap()

        self.state = state[..., numpy.newaxis]
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
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def update_(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _update_datamap(self):
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

class rankSTEDRecurrentMultiObjectivesWithArticulationEnv(gym.Env):
    """
    Creates a `rankSTEDRecurrentMultiObjectivesEnv`

    Action space
        The action space is a tuple, where
        1. The imaging parameter selection
        2. The action to choose from
            {0: Acquire image, 1: Ask for expert knowledge, 2: Final parameters}

    Observation space
        The observation space is a tuple, where
        1. The current confocal image
        2. A vector containing the current articulation, the selected actions, the obtained objectives

    """
    metadata = {'render.modes': ['human']}
    obj_names = ["Resolution", "Bleach", "SNR"]

    def __init__(self, bleach_sampling="constant", actions=["p_sted"],
                    max_num_requests=1, max_episode_steps=10, select_final=False,
                    scale_rank_reward=False, scale_nanodomain_reward=1.):

        self.actions = actions
        self.select_final = select_final
        self.scale_nanodomain_reward = scale_nanodomain_reward

        if self.select_final:
            self.action_space = spaces.Box(
                low=numpy.array([defaults.action_spaces[name]["low"] for name in self.actions] + [0]),
                high=numpy.array([defaults.action_spaces[name]["high"] for name in self.actions] + [2 + 1]),
                dtype=numpy.float32
            )
        else:
            self.action_space = spaces.Box(
                low=numpy.array([defaults.action_spaces[name]["low"] for name in self.actions] + [0]),
                high=numpy.array([defaults.action_spaces[name]["high"] for name in self.actions] + [1 + 1]),
                dtype=numpy.float32
            )

        self.observation_space = spaces.Tuple((
            spaces.Box(0, 2**16, shape=(64, 64, 1), dtype=numpy.uint16),
            spaces.Box(
                0, 1, shape=(max_episode_steps + len(self.obj_names) + len(self.actions),),
                dtype=numpy.float32
            ) # Articulation, shape is given by articulation vector, objectives, actions
        ))

        self.state = None
        self.initial_count = None
        self.synapse = None
        self.current_step = 0
        self.max_num_requests = max_num_requests
        self.num_request_left = max_num_requests
        self.bleach_sampling = bleach_sampling
        self.scale_rank_reward = scale_rank_reward
        self.current_articulation = -1
        self.episode_memory = {
            "actions" : [],
            "mo_objs" : [],
            "reward" : [],
        }

        self.datamap = None
        self.viewer = None

        self.seed()

        self.synapse_generator = SynapseGenerator(mode="mushroom", seed=None)
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

        # Loads preference articulation model
        self.preference_articulation = PreferenceArticulator()

    def step(self, action):

        # Action is an array of size self.actions and main_action
        # main action should be in the [0, 1, 2]
        # We manually clip the actions which are out of action space
        # We ensure that the action values are numbers
        action = numpy.nan_to_num(action)
        action = numpy.clip(action, self.action_space.low, self.action_space.high)
        if self.select_final:
            imaging_action, main_action = action[:len(self.actions)], min(int(action[-1]), 2)
        else:
            imaging_action, main_action = action[:len(self.actions)], min(int(action[-1]), 1)

        # On the last step of the environment we enforce the final decision
        if self.current_step >= self.spec.max_episode_steps - 1:
            main_action = 2

        if (main_action == 0) or (main_action == 1 and self.num_request_left <= 0):
            # Acquire an image with the given parameters
            sted_image, bleached, conf1, conf2, fg_s, fg_c = self._acquire(imaging_action)
            mo_objs = self.mo_reward_calculator.evaluate(sted_image, conf1, conf2, fg_s, fg_c)

            # Reward is proportionnal to the ranked position of the last image
            articulation, sorted_indices = self.preference_articulation.articulate(
                self.episode_memory["mo_objs"] + [mo_objs]
            )
            index = numpy.argmax(sorted_indices).item()
            # Reward is given by the position in the sorting
            reward = (index + 1) / len(sorted_indices)

            done = False

        elif main_action == 1:
            # Asking for preference
            sted_image = numpy.zeros(self.observation_space[0].shape[:-1])
            bleached = numpy.zeros(self.observation_space[0].shape[:-1])
            conf1 = numpy.zeros(self.observation_space[0].shape[:-1])
            conf2 = numpy.zeros(self.observation_space[0].shape[:-1])
            fg_s = numpy.zeros(self.observation_space[0].shape[:-1])
            fg_c = numpy.zeros(self.observation_space[0].shape[:-1])

            reward = 1
            mo_objs = [scales_dict[obj_name]["high"] if obj_name != "SNR" else scales_dict[obj_name]["low"] for obj_name in self.obj_names]
            self.num_request_left -= 1

            if len(self.episode_memory["mo_objs"]) > 0:
                self.current_articulation, _ = self.preference_articulation.articulate(self.episode_memory["mo_objs"])

            done = False

        elif main_action == 2:
            # Acquire final image
            sted_image, bleached, conf1, conf2, fg_s, fg_c = self._acquire(imaging_action)
            mo_objs = self.mo_reward_calculator.evaluate(sted_image, conf1, conf2, fg_s, fg_c)

            reward = self.nb_reward_calculator.evaluate(sted_image, conf1, conf2, fg_s, fg_c, synapse=self.synapse)
            reward = reward * self.scale_nanodomain_reward

            done = True

        else:
            raise NotImplementedError("Only 3 actions supported...")

        # Updates memory
        self.current_step += 1
        self.episode_memory["mo_objs"].append(mo_objs)
        self.episode_memory["actions"].append(imaging_action)
        self.episode_memory["reward"].append(reward)

        state = self._update_datamap()
        self.state = state[..., numpy.newaxis]

        info = {
            "action" : action,
            "bleached" : bleached,
            "sted_image" : sted_image,
            "conf1" : conf1,
            "conf2" : conf2,
            "fg_c" : fg_c,
            "fg_s" : fg_s,
            "mo_objs" : mo_objs,
            "articulation" : self.current_articulation,
            "reward" : reward
        }
        if self.current_articulation == -1:
            articulation = numpy.zeros((self.spec.max_episode_steps, ))
        else:
            articulation = numpy.eye(self.spec.max_episode_steps)[self.current_articulation]

        # Build the observation space
        obs = []
        obs.extend(articulation)
        obs.extend(self.episode_memory["actions"][-1])
        obs.extend(self.episode_memory["mo_objs"][-1])
        obs = numpy.array(obs)

        return (self.state, obs), reward, done, info

    def reset(self):
        """
        Resets the environment with a new datamap

        :returns : A `numpy.ndarray` of the molecules
        """
        # Updates the current bleach function
        self.microscope = self.microscope_generator.generate_microscope(
            phy_react=self.bleach_sampler.sample()
        )

        self.num_request_left = self.max_num_requests
        self.current_step = 0
        self.current_articulation = -1
        self.episode_memory = {
            "actions" : [],
            "mo_objs" : [],
            "reward" : [],
        }

        state = self._update_datamap()

        self.state = state[..., numpy.newaxis]
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
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def update_(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _update_datamap(self):
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

if __name__ == "__main__":

    import time

    from gym.envs.registration import EnvSpec

    env = rankSTEDMultiObjectivesWithDelayedRewardEnv(actions=["p_sted"], bleach_sampling="normal")
    env.spec = EnvSpec("STEDranking-v0", max_episode_steps=3)
    for i in range(10):
        obs = env.reset()

        while True:
            obs, reward, done, info = env.step(env.action_space.sample())
            # print(obs[1])
            # obs, reward, done, info = env.step(numpy.array(i))
            print(reward, info["mo_objs"], info["action"][-1])

            if done:
                print(f"Episode {i} done.\n")
                break
