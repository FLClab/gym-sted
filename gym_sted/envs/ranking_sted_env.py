
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
from gym_sted.utils import SynapseGenerator, MicroscopeGenerator, get_foreground
from gym_sted.rewards import objectives
from gym_sted.prefnet import PreferenceArticulator

obj_dict = {
    "SNR" : objectives.Signal_Ratio(75),
    "Bleach" : objectives.Bleach(),
    "Resolution" : objectives.Resolution(pixelsize=20e-9),
    "Squirrel" : objectives.Squirrel()
}
bounds_dict = {
    "SNR" : {"min" : 0.20, "max" : numpy.inf},
    "Bleach" : {"min" : -numpy.inf, "max" : 0.5},
    "Resolution" : {"min" : 0, "max" : 100}
}
scales_dict = {
    "SNR" : {"min" : 0, "max" : 1},
    "Bleach" : {"min" : 0, "max" : 1},
    "Resolution" : {"min" : 40, "max" : 180}
}
action_spaces = {
    "p_sted" : {"low" : 5.0e-6, "high" : 5.0e-3},
    "p_ex" : {"low" : 0.8e-6, "high" : 5.0e-6},
    "pdt" : {"low" : 10.0e-6, "high" : 150.0e-6},
}

class rankSTEDEnv(gym.Env):
    """
    Creates a `STEDEnv`

    Action space
        The action space here is a tuple, where
        1. The first element of the tuple represents the imaging parameter selection
        2. The second element of the tuple represents the action to choose from
            {0: Acquire image, 1: Ask for expert knowledge, 2: Final parameters}

    """
    metadata = {'render.modes': ['human']}
    obj_names = ["Resolution", "Bleach", "SNR"]

    def __init__(self, reward_calculator="SumRewardCalculator", actions=["p_sted"],
                    max_num_requests=1, max_episode_steps=10):

        self.synapse_generator = SynapseGenerator(mode="mushroom", seed=42)
        self.microscope_generator = MicroscopeGenerator()
        self.microscope = self.microscope_generator.generate_microscope()

        self.actions = actions
        self.action_space = spaces.Box(
            low=numpy.array([action_spaces[name]["low"] for name in self.actions] + [0]),
            high=numpy.array([action_spaces[name]["high"] for name in self.actions] + [2 + 1]),
            dtype=numpy.float32
        )

        self.observation_space = spaces.Tuple((
            spaces.Box(0, 2**16, shape=(64, 64, 1), dtype=numpy.uint16),
            spaces.Box(0, 1, shape=(max_episode_steps,), dtype=numpy.float32)
        ))

        self.state = None
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
            rewards = numpy.zeros((len(self.obj_names, )))
            self.num_request_left -= 1

            if len(self.cummulated_rewards["rewards"]) > 0:
                self.current_articulation, _ = self.preference_articulation.articulate(self.cummulated_rewards["rewards"])

        elif main_action == 2:
            # Acquire final image
            sted_image, bleached, conf1, conf2, fg_s, fg_c = self._acquire(imaging_action)
            reward = self.reward_calculator.evaluate(sted_image, conf1, conf2, fg_s, fg_c)
            rewards = self._reward_calculator.evaluate(sted_image, conf1, conf2, fg_s, fg_c)

            # Here reward should be 1 if reward is >= expert choice else it should be 0
            # We do not give a reward if all rewards are 0
            if (self.current_articulation != -1) and (reward > 0):
                r = int(reward >= self.cummulated_rewards["reward"][self.current_articulation])
                reward += r
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
        return [self.state, articulation], reward, done, info

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
        return [self.state, numpy.zeros((self.spec.max_episode_steps,))]

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
        molecules_disposition = self.synapse_generator()
        self.datamap = self.microscope_generator.generate_datamap(
            datamap = {
                "whole_datamap" : molecules_disposition,
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

    from gym.envs.registration import EnvSpec

    env = rankSTEDEnv(reward_calculator="BoundedRewardCalculator", actions=["p_sted"])
    env.spec = EnvSpec("STEDranking-v0", max_episode_steps=10)
    for i in range(10):
        obs = env.reset()

        while True:
            obs, reward, done, info = env.step(env.action_space.sample())
            # obs, reward, done, info = env.step(numpy.array(i))
            print(reward, info["rewards"], info["action"])
            if done:
                print(f"Episode {i} done.\n")
                break
