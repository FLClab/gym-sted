
import gym
import numpy
import random
import os
import queue

import pysted.base
from gym import error, spaces, utils
from gym.utils import seeding
from matplotlib import pyplot
from collections import OrderedDict

import gym_sted
from gym_sted import rewards, defaults
from gym_sted.utils import SynapseGenerator, MicroscopeGenerator, RecordingQueue, get_foreground
from gym_sted.rewards import objectives
from gym_sted.prefnet import PreferenceArticulator

# I will copy the values straight from Anthony's ranking_sted_env, need to think about what values
# I actually want to use

obj_dict = {
    "SNR" : objectives.Signal_Ratio(75),
    "Bleach" : objectives.Bleach(),
    "Resolution" : objectives.Resolution(pixelsize=20e-9),
    "Squirrel" : objectives.Squirrel(),
    "NbNanodomains" : objectives.NumberNanodomains()
}
bounds_dict = {
    "SNR" : {"min" : 0.20, "max" : numpy.inf},
    "Bleach" : {"min" : -numpy.inf, "max" : 0.5},
    "Resolution" : {"min" : 0, "max" : 100},
    "NbNanodomains" : {"min" : 0, "max" : numpy.inf}
}
scales_dict = {
    "SNR" : {"min" : 0, "max" : 1},
    "Bleach" : {"min" : 0, "max" : 1},
    "Resolution" : {"min" : 40, "max" : 180},
    "NbNanodomains" : {"min" : 0, "max" : 1}   # ???
}
action_spaces = {
    # changed p_sted low to 0 as I want to 0. as I want to take confocals if the flash is not yet happening
    "p_sted" : {"low" : 0., "high" : 5.0e-3},
    "p_ex" : {"low" : 0.8e-6, "high" : 5.0e-6},   # jveux tu lui laisser prendre un p_ex = 0 ? ferait la wait action...
    "pdt" : {"low" : 10.0e-6, "high" : 150.0e-6},
}

class timedExpSTEDEnv(gym.Env):
    """
    Creates a 'STEDEnv'

    Action space
        The action space here is a selection of the p_ex, p_sted and pdt values (?)
    """
    metadata = {'render.modes': ['human']}
    obj_names = ["Resolution", "Bleach", "SNR", "NbNanodomains"]

    def __init__(self, time_quantum_us=1, exp_time_us=500000, actions=["p_sted"],
                 reward_calculator="SumRewardCalculator"):
        self.synapse_generator = SynapseGenerator(mode="mushroom", n_nanodomains=7, n_molecs_in_domain=100, seed=42)
        self.microscope_generator = MicroscopeGenerator()
        self.microscope = self.microscope_generator.generate_microscope()

        self.actions = actions
        self.action_space = spaces.Box(
            low=numpy.array([action_spaces[name]["low"] for name in self.actions] + [0]),
            high=numpy.array([action_spaces[name]["high"] for name in self.actions] + [2 + 1]),
            dtype=numpy.float32
        )

        self.dmap_shape = (64, 64)
        self.q_length = 4
        self.max_episode_steps = int(numpy.ceil(exp_time_us /
                                                (self.dmap_shape[0] * self.dmap_shape[1] * action_spaces["pdt"]["low"] * 1e6)))
        # since this is a temporal experiment, I think it would be relevant to have the last 4 acquisitions as the
        # observation of the state. Need to figure out how to do a first in first out thing for this
        self.observation_space = spaces.Box(0, 2 ** 16, shape=(self.dmap_shape[0], self.dmap_shape[1], self.q_length),
                                            dtype=numpy.uint16)


        self.state = None
        self.initial_count = None

        self.cummulated_rewards = {
            "rewards": [],
            "reward": []
        }

        objs = OrderedDict({obj_name: obj_dict[obj_name] for obj_name in self.obj_names})
        bounds = OrderedDict({obj_name: bounds_dict[obj_name] for obj_name in self.obj_names})
        scales = OrderedDict({obj_name: scales_dict[obj_name] for obj_name in self.obj_names})
        self.reward_calculator = getattr(rewards, reward_calculator)(objs, bounds=bounds, scales=scales)
        self._reward_calculator = rewards.MORewardCalculator(objs, bounds=bounds, scales=scales)

        self.temporal_datamap = None
        self.viewer = None

        self.time_quantum_us = time_quantum_us
        self.exp_time_us = exp_time_us

        self.clock = None
        self.temporal_experiment = None

        self.seed()

    def step(self, action):
        pass

    def reset(self):
        synapse = self.synapse_generator.generate()

        self.temporal_datamap = self.microscope_generator.generate_temporal_datamap(
            temporal_datamap = {
                "whole_datamap" : synapse.frame,
                "datamap_pixelsize" : self.microscope_generator.pixelsize,
                "synapse_obj": synapse
            }
        )

        conf_params = self.microscope_generator.generate_params()
        first_acq, _, _ = self.microscope.get_signal_and_bleach(
            self.temporal_datamap, self.temporal_datamap.pixelsize, **conf_params, bleach=False
        )
        # for the state, fill a queue with the first observation,
        # If I only add it once to the Q the head will point to an array of zeros until the Q is filled,
        # which is not what I want
        start_data = []
        for i in range(self.q_length):
            start_data.append(first_acq)
        start_data = numpy.array(start_data)
        self.state = RecordingQueue(start_data, maxlen=self.q_length, num_sensors=self.dmap_shape)
        # là mon state c'est un Q object, est-ce que c'est good? pour les passer à mes nn? idk?

        self.temporal_experiment = pysted.base.TemporalExperiment(self.clock, self.microscope, self.temporal_datamap,
                                                                  self.exp_time_us, bleach=True)

        return self.state.to_array()   # I think this is how I need to return it to ensure it can go through the nn




    def render(self, info, mode="'human"):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        return None



if __name__ == "__main__":
    xd = timedExpSTEDEnv()
    xd.reset()