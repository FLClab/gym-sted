
import gym
import numpy
import random
import os
import abberior
import yaml

from gym import error, spaces, utils
from gym.utils import seeding
from matplotlib import pyplot
from collections import OrderedDict, defaultdict

import gym_sted
from gym_sted import rewards, defaults
from gym_sted.utils import SynapseGenerator, MicroscopeGenerator, get_foreground, BleachSampler, Normalizer
from gym_sted.rewards import objectives
from gym_sted.prefnet import PreferenceArticulator, load_demonstrations
from gym_sted.defaults import obj_dict, bounds_dict, scales_dict

from gym_sted.microscopes.abberior import AbberiorMicroscope

# Requires stedopt to be installed
from stedopt.tools import RegionSelector

class AbberiorSTEDMultiObjectivesEnv(gym.Env):
    """
    Creates a `AbberiorSTEDMultiObjectivesEnv`

    Action space
        The action space corresponds to the imaging parameters

    Observation space
        The observation space is a tuple, where
        1. The current confocal image, and previous confocal/STED images
        2. A vector containing the selected actions, the obtained objectives

    """
    metadata = {'render.modes': ['human']}
    obj_names = ["Resolution", "Bleach", "SNR"]

    def __init__(self, actions=["p_sted", "p_ex", "pdt"],
                    max_episode_steps=30,
                    normalize_observations=True):

        # Sets the configuration
        self.config_overview = abberior.microscope.get_config("Setting overview configuration.")
        self.config_conf = abberior.microscope.get_config("Setting confocal configuration.")
        self.config_sted = abberior.microscope.get_config("Setting STED configuration.")
        self.config_focus = abberior.microscope.get_config("Setting FOCUS configuration.")

        self.actions = actions
        self.default_action_space = defaults.abberior_action_spaces.copy()

        self.action_space = spaces.Box(
            low=numpy.array([self.default_action_space[name]["low"] for name in self.actions]),
            high=numpy.array([self.default_action_space[name]["high"] for name in self.actions]),
            dtype=numpy.float32
        )

        # The observation space is a tuple of the current confocal image and the articulation
        # The shape of the image must match the shape of the acquisition
        self.observation_space = spaces.Tuple((
            spaces.Box(0, 2**16, shape=(224, 224, 3), dtype=numpy.uint16),
            spaces.Box(
                0, 1024, shape=(len(self.obj_names) * max_episode_steps + len(self.actions) * max_episode_steps,),
                dtype=numpy.float32
            ) # Articulation, shape is given by objectives, actions at each steps
        ))

        self.state = None
        self.current_step = 0
        self.episode_memory = {
            "actions" : [],
            "mo_objs" : [],
            "reward" : [],
        }

        self.viewer = None

        self.measurements = {
            "conf" : self.config_conf,
            "sted" : self.config_sted,
            "focus" : self.config_focus
        }
        self.microscope = AbberiorMicroscope(
            self.measurements
        )

        objs = OrderedDict({obj_name : obj_dict[obj_name] for obj_name in self.obj_names})
        bounds = OrderedDict({obj_name : bounds_dict[obj_name] for obj_name in self.obj_names})
        scales = OrderedDict({obj_name : scales_dict[obj_name] for obj_name in self.obj_names})
        self.mo_reward_calculator = rewards.MORewardCalculator(objs, bounds=bounds, scales=scales)

        # Loads preference articulation model
        self.preference_articulation = PreferenceArticulator()

        # Creates an action and objective normalizer
        self.normalize_observations = normalize_observations
        self.action_normalizer = Normalizer(self.actions, self.default_action_space)
        self.obj_normalizer = Normalizer(self.obj_names, scales_dict)

        self.region_selector = RegionSelector(
            self.config_overview, {
                "region_opts" : {
                    "mode" : "manual",
                    "overview" : "640"
                }
            })

    def step(self, action):
        """
        Performs a step in the environment

        :param action: A `numpy.ndarray` of the action
        """

        # Action is an array of size self.actions and main_action
        # main action should be in the [0, 1, 2]
        # We manually clip the actions which are out of action space
        action = numpy.clip(action, self.action_space.low, self.action_space.high)

        # Acquire an image with the given parameters
        sted_image, conf1, conf2, fg_s, fg_c = self._acquire(action)
        mo_objs = self.mo_reward_calculator.evaluate(sted_image, conf1, conf2, fg_s, fg_c)

        # Reward is given by the objectives
        reward, _, _ = self.preference_articulation.articulate(
            [mo_objs]
        )
        reward = reward.item()

        # Updates memory
        done = self.current_step >= self.spec.max_episode_steps - 1
        self.current_step += 1
        self.episode_memory["mo_objs"].append(mo_objs)
        self.episode_memory["actions"].append(action)
        self.episode_memory["reward"].append(reward)

        info = {
            "action" : action,
            "sted_image" : sted_image,
            "conf1" : conf1,
            "conf2" : conf2,
            "fg_c" : fg_c,
            "fg_s" : fg_s,
            "mo_objs" : mo_objs,
            "reward" : reward,
        }

        # Build the observation space
        obs = []
        for a, mo in zip(self.episode_memory["actions"], self.episode_memory["mo_objs"]):
            obs.extend(self.action_normalizer(a) if self.normalize_observations else a)
            obs.extend(self.obj_normalizer(mo) if self.normalize_observations else mo)
        obs = numpy.pad(numpy.array(obs), (0, self.observation_space[1].shape[0] - len(obs)))

        state = self._update_datamap()

        self.state = numpy.stack((state, conf1, sted_image), axis=-1)

        return (self.state.astype(numpy.uint16), obs.astype(numpy.float32)), reward, done, False, info


    def reset(self, seed=None, options=None):
        """
        Resets the environment with a new state

        :returns : The current state of the environment
                   A `dict` of information
        """
        super().reset(seed=seed)

        self.current_step = 0
        self.episode_memory = {
            "actions" : [],
            "mo_objs" : [],
            "reward" : [],
        }

        state = self._update_datamap()

        self.state = numpy.stack((state, numpy.zeros_like(state), numpy.zeros_like(state)), axis=-1)
        return (self.state.astype(numpy.uint16), numpy.zeros((self.observation_space[1].shape[0],), dtype=numpy.float32)), {}

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

    def update_(self, **kwargs):
        """
        Updates parameters of the environment in place.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _update_datamap(self):
        """
        Updates the state of the microscope. 

        This corresponds to the acquisition of a new image at a new position. The user is prompted to move the stage to a new position if necessary. The `RegionSelector` is used to keep track of the selected regions by the user.
        """
        # Sets the next regions to images
        xoff, yoff = next(self.region_selector)
        abberior.microscope.set_offsets(self.measurements["conf"], xoff, yoff)
        abberior.microscope.set_offsets(self.measurements["sted"], xoff, yoff)
        abberior.microscope.set_offsets(self.measurements["focus"], xoff, yoff)

        input("Now is a good time to move focus. Press enter when done.")

        state = self.microscope.acquire("conf")

        return state

    def acquire(self, action):
        return self._acquire(action)

    def _acquire(self, action):
        """
        Acquires an image with the given action.

        The sequence of acquisition is as follows:
        1. Acquire a confocal image
        2. Acquire a STED image
        3. Acquire a second confocal image
        4. Calculate the foreground of the confocal/STED images

        :param action: A `numpy.ndarray` of the action

        :return: A `numpy.ndarray` of the STED image
                 A `numpy.ndarray` of the first confocal image
                 A `numpy.ndarray` of the second confocal image
                 A `numpy.ndarray` of the STED foreground
                 A `numpy.ndarray` of the confocal foreground
        """
        # Generates imaging parameters
        sted_params = {
            name : action[self.actions.index(name)]
                for name in ["pdt", "p_ex", "p_sted"]
                if name in self.actions
        }

        # Acquire confocal image
        conf1 = self.microscope.acquire("conf")

        # Acquire STED image
        sted_image = self.microscope.acquire("sted", sted_params)

        # Acquire confocal image
        conf2 = self.microscope.acquire("conf")

        # foreground on confocal image
        fg_c = get_foreground(conf1)
        # foreground on sted image
        if numpy.any(sted_image):
            fg_s = get_foreground(sted_image)
        else:
            fg_s = numpy.ones_like(fg_c)
        # remove STED foreground points not in confocal foreground, if any
        fg_s *= fg_c

        return sted_image, conf1, conf2, fg_s, fg_c

    def get_state(self):
        """
        Returns a `dict` of the state of the `env`
        """
        state = {
            "action_space" : self.default_action_space,
            "scales_dict" : scales_dict,
            "microscope-defaults" : self.microscope.config
        }
        return state

    def close(self):
        return None

class AbberiorSTEDCountRateMultiObjectivesEnv(AbberiorSTEDMultiObjectivesEnv):
    """
    Creates a `AbberiorSTEDMultiObjectivesEnv`

    Action space
        The action space corresponds to the imaging parameters

    Observation space
        The observation space is a tuple, where
        1. The current confocal image, and previous confocal/STED images
        2. A vector containing the the confocal power, the selected actions, the obtained objectives

    """
    metadata = {'render.modes': ['human']}
    obj_names = ["Resolution", "Bleach", "SNR"]

    def __init__(self, actions=["p_sted", "p_ex", "pdt"],
                    max_episode_steps=30,
                    normalize_observations=True,
                    max_count_rate=20e+6, 
                    negative_reward=-10):
        """
        Instantiates the `AbberiorSTEDCountRateMultiObjectivesEnv`

        :param actions: A `list` of the actions
        :param max_episode_steps: An `int` of the maximum number of steps
        :param normalize_observations: A `bool` that specifies if the observations should be normalized
        :param max_count_rate: A `float` of the maximum count rate allowed for a STED acquisition
        :param negative_reward: A `float` of the negative reward given when the count rate is above the maximum
        """
        # These are the parameters by default that were used in the simulated environment
        # These parameters should be updated to match the real microscope
        self.conf_params = {
            "p_ex" : 9,
            "p_sted" : 0.,
            "pdt" : 10e-6,
        }

        self.negative_reward = negative_reward
        self.max_count_rate = max_count_rate

        super(AbberiorSTEDCountRateMultiObjectivesEnv, self).__init__(actions, max_episode_steps, normalize_observations)

        self.observation_space = spaces.Tuple((
            spaces.Box(0, 2**16, shape=(224, 224, 3), dtype=numpy.uint16),
            spaces.Box(
                0, 1, shape=(1 + len(self.obj_names) * max_episode_steps + len(self.actions) * max_episode_steps,),
                dtype=numpy.float32
            ) # Articulation, shape is given by objectives, actions at each steps
        ))

    def step(self, action):
        """
        Performs a step in the environment

        :param action: A `numpy.ndarray` of the action
        """

        # Action is an array of size self.actions and main_action
        # main action should be in the [0, 1, 2]
        # We manually clip the actions which are out of action space
        action = numpy.clip(action, self.action_space.low, self.action_space.high)

        # Acquire an image with the given parameters
        sted_image, conf1, conf2, fg_s, fg_c = self._acquire(action)
        mo_objs = self.mo_reward_calculator.evaluate(sted_image, conf1, conf2, fg_s, fg_c)

        # Reward is given by the objectives
        reward, _, _ = self.preference_articulation.articulate(
            [mo_objs]
        )
        reward = reward.item()

        # Generates imaging parameters
        sted_params = {
            name : action[self.actions.index(name)]
                for name in ["pdt", "p_ex", "p_sted"]
                if name in self.actions
        }
        count_rate = sted_image.max() / sted_params["pdt"]
        if count_rate > self.max_count_rate:
            reward = self.negative_reward        

        # Updates memory
        done = self.current_step >= self.spec.max_episode_steps - 1
        self.current_step += 1
        self.episode_memory["mo_objs"].append(mo_objs)
        self.episode_memory["actions"].append(action)
        self.episode_memory["reward"].append(reward)

        info = {
            "action" : action,
            "sted_image" : sted_image,
            "conf1" : conf1,
            "conf2" : conf2,
            "fg_c" : fg_c,
            "fg_s" : fg_s,
            "mo_objs" : mo_objs,
            "reward" : reward,
        }

        # Build the observation space
        conf_params = self.microscope.config["params_conf"]
        obs = [conf_params["p_ex"] / self.conf_params["p_ex"]]
        for a, mo in zip(self.episode_memory["actions"], self.episode_memory["mo_objs"]):
            obs.extend(self.action_normalizer(a) if self.normalize_observations else a)
            obs.extend(self.obj_normalizer(mo) if self.normalize_observations else mo)
        obs = numpy.pad(numpy.array(obs), (0, self.observation_space[1].shape[0] - len(obs)))

        state = self._update_datamap()

        self.state = numpy.stack((state, conf1, sted_image), axis=-1)

        return (self.state.astype(numpy.uint16), obs.astype(numpy.float32)), reward, done, False, info

class MetaEnv:
    
    def __init__(self, envs):

        self.envs = envs
        self.action_space = spaces.Tuple([env.action_space for env in envs])
        self.observation_space = spaces.Tuple([env.observation_space for env in envs])

        self.max_episode_steps = max([env.spec.max_episode_steps for env in envs])
        self.current_step = 0
        self.episode_memory = {
            "actions" : [],
            "mo_objs" : [],
            "reward" : [],
        }

        self.state = None
        self.viewer = None

        self.config_overview = abberior.microscope.get_config("Setting overview configuration.")
        self.config_focus = abberior.microscope.get_config("Setting FOCUS configuration.")
        self.region_selector = RegionSelector(
            self.config_overview,
            {
                "region_opts" : {
                    "mode" : "random",
                    "overview" : "640"
                }
            }
        )

    def step(self, actions):
        """
        Performs a step in the environment

        :param action: A `numpy.ndarray` of the action
        """

        # Action is an array of size self.actions and main_action
        # main action should be in the [0, 1, 2]
        # We manually clip the actions which are out of action space
        actions = [numpy.clip(action, env.action_space.low, env.action_space.high) for action, env in zip(actions, self.envs)]

        # Acquire an image with the given parameters
        sted_images = []
        conf1s = []
        conf2s = []
        fg_ss = []
        fg_cs = []
        mo_objs = []
        rewards = []
        for action, env in zip(actions, self.envs):
            sted_image, conf1, conf2, fg_s, fg_c = env.acquire(action)
            sted_images.append(sted_image)
            conf1s.append(conf1)
            conf2s.append(conf2)
            fg_ss.append(fg_s)
            fg_cs.append(fg_c)

            mo_obj = env.mo_reward_calculator.evaluate(sted_image, conf1, conf2, fg_s, fg_c)
            mo_objs.append(mo_obj)

            reward, _, _ = env.preference_articulation.articulate(
                [mo_obj]
            )
            rewards.append(reward.item())

        # Updates memory
        done = self.current_step >= self.max_episode_steps - 1
        self.current_step += 1
        self.episode_memory["mo_objs"].append(mo_objs)
        self.episode_memory["actions"].append(actions)
        self.episode_memory["reward"].append(rewards)
        info = {
            "actions" : actions,
            "sted_images" : sted_images,
            "conf1s" : conf1s,
            "conf2s" : conf2s,
            "fg_cs" : fg_cs,
            "fg_ss" : fg_ss,
            "mo_objs" : mo_objs,
            "rewards" : rewards,
        }

        # Build the observation space
        obs_per_env = []
        for env in self.envs:
            conf_params = env.microscope.config["params_conf"]
            obs = [conf_params["p_ex"] / env.conf_params["p_ex"]]
            for actions, mos in zip(self.episode_memory["actions"], self.episode_memory["mo_objs"]):
                a = actions[self.envs.index(env)]
                mo = mos[self.envs.index(env)]

                obs.extend(env.action_normalizer(a) if env.normalize_observations else a)
                obs.extend(env.obj_normalizer(mo) if env.normalize_observations else mo)

            obs = numpy.pad(numpy.array(obs), (0, env.observation_space[1].shape[0] - len(obs)))
            obs_per_env.append(obs)
        
        states = self._update_datamap(envs=self.envs)
        state_per_env = []
        for state, conf1, sted_image in zip(states, conf1s, sted_images):
            stack = numpy.stack((state, conf1, sted_image), axis=-1)
            stack = stack / 2**10 # This is the normalization used in the original code
            state_per_env.append(stack)
        
        return (state_per_env, obs_per_env), rewards, done, False, info

    def reset(self, seed=None, options=None):
        """
        Resets the environment with a new state

        :returns : The current state of the environment
                   A `dict` of information
        """

        self.current_step = 0
        self.episode_memory = {
            "actions" : [],
            "mo_objs" : [],
            "reward" : [],
        }

        states = self._update_datamap(envs=self.envs)
        state_per_env = []
        for state in states:
            stack = numpy.stack((state, numpy.zeros_like(state), numpy.zeros_like(state)), axis=-1)
            stack = stack / 2**10 # This is the normalization used in the original code
            state_per_env.append(stack)
        return (state_per_env, [numpy.zeros((env.observation_space[1].shape[0],), dtype=numpy.float32) for env in self.envs])

    def get_state(self):
        """
        Returns a `dict` of the state of the `env`
        """
        states = defaultdict(list)
        for env in self.envs:
            state = env.get_state()
            for key, value in state.items():
                states[key].append(value)
        return states

    def close(self):
        return None

    def _update_datamap(self, envs):
        """
        Updates the state of the microscope. 

        This corresponds to the acquisition of a new image at a new position. The user is prompted to move the stage to a new position if necessary. The `RegionSelector` is used to keep track of the selected regions by the user.
        """
        # Sets the next regions to images
        xoff, yoff = next(self.region_selector)

        abberior.microscope.set_offsets(self.config_focus, xoff, yoff)
        input("Now is a good time to move focus. Press enter when done.")

        states = []
        for env in envs:
            abberior.microscope.set_offsets(env.measurements["conf"], xoff, yoff)
            abberior.microscope.set_offsets(env.measurements["sted"], xoff, yoff)
        
            state = env.microscope.acquire("conf")
            states.append(state)
        return states