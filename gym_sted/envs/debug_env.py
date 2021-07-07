import gym
import numpy
import random

from gym import error, spaces, utils
from gym.utils import seeding
from matplotlib import pyplot
from collections import OrderedDict

# from gym_sted.utils import SynapseGenerator, MicroscopeGenerator, get_foreground
from gym_sted.utils import MicroscopeGenerator, get_foreground
# from gym_sted.rewards import BoundedRewardCalculator, RewardCalculator, objectives
from gym_sted.rewards import objectives

obj_dict = {
    "SNR" : objectives.Signal_Ratio(75),
    "Bleach" : objectives.Bleach(),
    "Resolution" : objectives.Resolution(pixelsize=20e-9),
    "Squirrel" : objectives.Squirrel()
}
bounds_dict = {
    "SNR" : {"min" : 0.20, "max" : numpy.inf},
    "Bleach" : {"min" : -numpy.inf, "max" : 0.5},
    "Resolution" : {"min" : 0, "max" : 80}
}

class DebugBleachSTEDEnv(gym.Env):

    obj_names = ["Resolution", "Bleach", "SNR"]

    def __init__(self):

        self.synapse_generator = SynapseGenerator(mode="mushroom", seed=42)
        self.microscope_generator = MicroscopeGenerator()
        self.microscope = self.microscope_generator.generate_microscope()

        self.action_space = spaces.Box(low=0., high=5e-3, shape=(1,), dtype=numpy.float32)
        self.observation_space = spaces.Box(0, 2**16, shape=(1, 20, 20), dtype=numpy.uint16)

        self.state = None
        self.initial_count = None

        objs = OrderedDict({obj_name : obj_dict[obj_name] for obj_name in self.obj_names})
        bounds = OrderedDict({obj_name : bounds_dict[obj_name] for obj_name in self.obj_names})
        self.reward_calculator = BoundedRewardCalculator(objs, bounds)
        # self._reward_calculator = RewardCalculator(objs)

        self.datamap = None
        self.viewer = None

        self.seed()

    def step(self, action):

        # We manually rescale and clip the actions which are out of action space
        m, M = -1, 1
        action = (action - m) / (M - m)
        action = action * (self.action_space.high - self.action_space.low) + self.action_space.low
        action = numpy.clip(action, self.action_space.low, self.action_space.high)

        # Generates imaging parameters
        sted_params = self.microscope_generator.generate_params(
            imaging = {
                "pdt" : 100.0e-6,
                "p_ex" : 2.0e-6,
                "p_sted" : action[0]
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

        # Only calculates the bleach
        c1, c2 = self.state.mean(), conf2.mean()
        reward = (c1 - c2) / c1

        # print(self._reward_calculator.evaluate(sted_image, conf1, conf2, fg_s, fg_c))
        # print(reward)

        done = conf2.sum() < 1.
        # print("EHY", done, reward)
        observation = conf2[numpy.newaxis, ...]
        info = {
            "bleached" : bleached,
            "sted_image" : sted_image,
            "conf1" : conf1,
            "conf2" : conf2,
            "fg_c" : fg_c,
            "fg_s" : fg_s
        }

        return observation, reward, done, info

    def reset(self):
        """
        Resets the environment with a new datamap
        :returns : A `numpy.ndarray` of the molecules
        """
        molecules_disposition = numpy.zeros((20, 20))
        molecules_disposition[5, 5] = 10
        self.datamap = self.microscope_generator.generate_datamap(
            datamap = {
                "whole_datamap" : molecules_disposition,
                "datamap_pixelsize" : self.microscope_generator.pixelsize
            }
        )

        # Acquire confocal image which sets the current state
        conf_params = self.microscope_generator.generate_params()
        self.state, _, _ = self.microscope.get_signal_and_bleach(
            self.datamap, self.datamap.pixelsize, **conf_params, bleach=False
        )

        self.initial_count = molecules_disposition.sum()
        return self.state[numpy.newaxis, ...]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        return None

class DebugResolutionSNRSTEDEnv(gym.Env):

    obj_names = ["Resolution", "SNR"]

    def __init__(self):

        self.synapse_generator = SynapseGenerator(mode="mushroom", seed=42)
        self.microscope_generator = MicroscopeGenerator()
        self.microscope = self.microscope_generator.generate_microscope()

        self.action_space = spaces.Box(low=0., high=5e-3, shape=(1,), dtype=numpy.float32)
        self.observation_space = spaces.Box(0, 2**16, shape=(1, 20, 20), dtype=numpy.uint16)

        self.state = None
        self.initial_count = None

        objs = OrderedDict({obj_name : obj_dict[obj_name] for obj_name in self.obj_names})
        bounds = OrderedDict({obj_name : bounds_dict[obj_name] for obj_name in self.obj_names})
        self.reward_calculator = RewardCalculator(objs)

        self.datamap = None
        self.viewer = None

        self.seed()

    def step(self, action):

        # We manually rescale and clip the actions which are out of action space
        m, M = -1, 1
        action = (action - m) / (M - m)
        action = action * (self.action_space.high - self.action_space.low) + self.action_space.low
        action = numpy.clip(action, self.action_space.low, self.action_space.high)

        # Generates imaging parameters
        sted_params = self.microscope_generator.generate_params(
            imaging = {
                "pdt" : 100.0e-6,
                "p_ex" : 2.0e-6,
                "p_sted" : action[0]
            }
        )
        conf_params = self.microscope_generator.generate_params()

        # Acquire confocal image
        conf1, bleached, _ = self.microscope.get_signal_and_bleach(
            self.datamap, self.datamap.pixelsize, **conf_params, bleach=False
        )

        # Acquire STED image
        sted_image, bleached, _ = self.microscope.get_signal_and_bleach(
            self.datamap, self.datamap.pixelsize, **sted_params, bleach=False
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

        reward = self.reward_calculator.evaluate(sted_image, conf1, conf2, fg_s, fg_c)
        print(reward)
        reward = (1 - (reward[0] - 40) / (250 - 40)) + reward[1]

        done = action == self.action_space.high
        # print("EHY", done, reward)
        observation = conf2[numpy.newaxis, ...]
        info = {
            "bleached" : bleached,
            "sted_image" : sted_image,
            "conf1" : conf1,
            "conf2" : conf2,
            "fg_c" : fg_c,
            "fg_s" : fg_s
        }

        return observation, reward, done, info

    def reset(self):
        """
        Resets the environment with a new datamap
        :returns : A `numpy.ndarray` of the molecules
        """
        molecules_disposition = self.synapse_generator()
        self.datamap = self.microscope_generator.generate_datamap(
            datamap = {
                "whole_datamap" : molecules_disposition,
                "datamap_pixelsize" : self.microscope_generator.pixelsize
            }
        )

        # Acquire confocal image which sets the current state
        conf_params = self.microscope_generator.generate_params()
        self.state, _, _ = self.microscope.get_signal_and_bleach(
            self.datamap, self.datamap.pixelsize, **conf_params, bleach=False
        )

        self.initial_count = molecules_disposition.sum()
        return self.state[numpy.newaxis, ...]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        return None
