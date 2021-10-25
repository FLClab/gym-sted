
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
from gym_sted.utils import SynapseGenerator, MicroscopeGenerator, RecordingQueue, get_foreground, BleachSampler
from gym_sted.rewards import objectives_timed, rewards_timed
from gym_sted.prefnet import PreferenceArticulator

obj_dict = {
    "SNR" : objectives_timed.Signal_Ratio(75),
    "Bleach" : objectives_timed.Bleach(),
    "Resolution" : objectives_timed.Resolution(pixelsize=20e-9),
    "NbNanodomains" : objectives_timed.NumberNanodomains()
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
action_spaces = {"low": 0., "high": 1.}


class preTrajDebugEnv(gym.Env):
    """
    The goal of this env is to see if my implementation of pre-traj training actually has an effect on the
    actions selected after

    The agent will be only 1 param. The agent can select a value between 0 - 1. The reward is 1 if the selected param
    is 1, 0 else. (maybe I could modify this to give the reward value as the param straight up?)

    An episode only lasts 1 step :)
    """

    def __init__(self):
        self.state = None

        self.action_space = spaces.Box(
            # low=numpy.array(action_spaces["low"]),
            # high=numpy.array(action_spaces["high"]),
            low=action_spaces["low"],
            high=action_spaces["high"],
            shape=(1,),
            dtype=numpy.float32
        )

        self.observation_space = spaces.Tuple((
            spaces.Box(0, 2 ** 16, shape=(64, 64, 1), dtype=numpy.uint16),
            spaces.Box(0, 1, shape=(1,), dtype=numpy.float32)
        ))

        self.max_episode_steps = 1

        self.seed()


    def step(self, action):
        action = numpy.clip(action, self.action_space.low, self.action_space.high)

        if action == 1:
            reward = 1
        else:
            reward = 0

        done = True
        observation = [numpy.zeros((64, 64, 1)), numpy.array([0])]

        info = {
            "action": action,
            "reward": reward
        }

        return observation, reward, done, info

    def reset(self):
        # ? dans mon exemple niaiseux y'a pas vrm d'obs jveux juste qu'il dise 1 :)
        return [numpy.zeros((64, 64, 1)), numpy.array([0])]

    def render(self, info, mode="human"):
        # ? jpense que j'ai rien à implem :)
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def update_(self, **kwargs):
        # unsure when this will be used
        for key, value in kwargs.items():
            setattr(self, key, value)

    def close(self):
        return None


class preTrajDebugEnvLin(gym.Env):
    """
    The goal of this env is to see if my implementation of pre-traj training actually has an effect on the
    actions selected after

    The agent will be only 1 param. The agent can select a value between 0 - 1. The reward is 1 if the selected param
    is 1, 0 else. (maybe I could modify this to give the reward value as the param straight up?)

    An episode only lasts 1 step :)
    """

    def __init__(self):
        self.state = None

        self.action_space = spaces.Box(
            # low=numpy.array(action_spaces["low"]),
            # high=numpy.array(action_spaces["high"]),
            low=action_spaces["low"],
            high=action_spaces["high"],
            shape=(1,),
            dtype=numpy.float32
        )

        self.observation_space = spaces.Tuple((
            spaces.Box(0, 2 ** 16, shape=(64, 64, 1), dtype=numpy.uint16),
            spaces.Box(0, 1, shape=(1,), dtype=numpy.float32)
        ))

        self.max_episode_steps = 1

        self.seed()


    def step(self, action):
        action = numpy.clip(action, self.action_space.low, self.action_space.high)

        reward = action[0]

        done = True
        observation = [numpy.zeros((64, 64, 1)), numpy.array([0])]

        info = {
            "action": action,
            "reward": reward
        }

        return observation, reward, done, info

    def reset(self):
        # ? dans mon exemple niaiseux y'a pas vrm d'obs jveux juste qu'il dise 1 :)
        return [numpy.zeros((64, 64, 1)), numpy.array([0])]

    def render(self, info, mode="human"):
        # ? jpense que j'ai rien à implem :)
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def update_(self, **kwargs):
        # unsure when this will be used
        for key, value in kwargs.items():
            setattr(self, key, value)

    def close(self):
        return None


class preTrajDebugEnvExp(gym.Env):
    """
    The goal of this env is to see if my implementation of pre-traj training actually has an effect on the
    actions selected after

    The agent will be only 1 param. The agent can select a value between 0 - 1. The reward is 1 if the selected param
    is 1, 0 else. (maybe I could modify this to give the reward value as the param straight up?)

    An episode only lasts 1 step :)
    """

    def __init__(self):
        self.state = None

        self.action_space = spaces.Box(
            # low=numpy.array(action_spaces["low"]),
            # high=numpy.array(action_spaces["high"]),
            low=action_spaces["low"],
            high=action_spaces["high"],
            shape=(1,),
            dtype=numpy.float32
        )

        self.observation_space = spaces.Tuple((
            spaces.Box(0, 2 ** 16, shape=(64, 64, 1), dtype=numpy.uint16),
            spaces.Box(0, 1, shape=(1,), dtype=numpy.float32)
        ))

        self.max_episode_steps = 1

        self.seed()


    def step(self, action):
        action = numpy.clip(action, self.action_space.low, self.action_space.high)

        reward = numpy.exp(numpy.log(2) * action[0]) - 1

        done = True
        observation = [numpy.zeros((64, 64, 1)), numpy.array([0])]

        info = {
            "action": action,
            "reward": reward
        }

        return observation, reward, done, info

    def reset(self):
        # ? dans mon exemple niaiseux y'a pas vrm d'obs jveux juste qu'il dise 1 :)
        return [numpy.zeros((64, 64, 1)), numpy.array([0])]

    def render(self, info, mode="human"):
        # ? jpense que j'ai rien à implem :)
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def update_(self, **kwargs):
        # unsure when this will be used
        for key, value in kwargs.items():
            setattr(self, key, value)

    def close(self):
        return None


if __name__ == "__main__":
     env = preTrajDebugEnvExp()

     state = env.reset()

     done = False
     while not done:
         print("stepping!")
         obs, r, done, info = env.step([0.9])
         print(f"reward = {r}")
     print("done stepping!")

