
import gym
import numpy
import random
import os
import abberior

from gym import error, spaces, utils
from gym.utils import seeding
from matplotlib import pyplot
from collections import OrderedDict

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
        1. The current confocal image
        2. A vector containing the current articulation, the selected actions, the obtained objectives

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

        self.actions = actions
        self.default_action_space = {
            "p_sted" : {"low" : 0., "high" : 50.},
            "p_ex" : {"low" : 0., "high" : 10.},
            "pdt" : {"low" : 1.0e-6, "high" : 30.0e-6},
        }

        self.action_space = spaces.Box(
            low=numpy.array([self.default_action_space[name]["low"] for name in self.actions]),
            high=numpy.array([self.default_action_space[name]["high"] for name in self.actions]),
            dtype=numpy.float32
        )

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
            "sted" : self.config_sted
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

        self.region_selector = RegionSelector(self.config_overview)

    def step(self, action):
        """
        Method that should be implemented in the object that inherited

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
        Resets the environment with a new datamap

        :returns : A `numpy.ndarray` of the molecules
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
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _update_datamap(self):

        # Sets the next regions to images
        xoff, yoff = next(self.region_selector)
        abberior.microscope.set_offsets(self.measurements["conf"], xoff, yoff)
        abberior.microscope.set_offsets(self.measurements["sted"], xoff, yoff)        

        state = self.microscope.acquire("conf")

        return state

    def _acquire(self, action):

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

    def close(self):
        return None
