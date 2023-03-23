
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
from gym_sted.prefnet import PreferenceArticulator, load_demonstrations
from gym_sted.defaults import obj_dict, bounds_dict, scales_dict

from .mosted_env import STEDMultiObjectivesEnv

class SequenceSTEDMultiObjectivesEnv(STEDMultiObjectivesEnv):
    """
    Creates a `SequenceSTEDMultiObjectivesEnv`

    Action space
        The action space corresponds to the imaging parameters

    Observation space
        The observation space is a tuple, where
        1. The current confocal image
        2. A vector containing the current articulation, the selected actions, the obtained objectives

    Goal
        This environment simulates an acquisition sequence on the same structure
        thus requiring the agent to tradeoff between high resolution image and
        photobleaching.
    """
    metadata = {'render.modes': ['human']}
    obj_names = ["Resolution", "Bleach", "SNR"]

    def __init__(self, bleach_sampling="constant", actions=["p_sted"],
                    max_episode_steps=10, scale_nanodomain_reward=1.,
                    normalize_observations=False):

        super(SequenceSTEDMultiObjectivesEnv, self).__init__(
            bleach_sampling = bleach_sampling,
            actions = actions,
            max_episode_steps = max_episode_steps,
            scale_nanodomain_reward = scale_nanodomain_reward,
            normalize_observations=normalize_observations
        )

    def reset(self, seed=None, options=None):
        """
        Resets the environment with a new datamap

        :returns : A `numpy.ndarray` of the molecules
        """
        self.synapse = self.synapse_generator(rotate=True)
        self.datamap = self.microscope_generator.generate_datamap(
            datamap = {
                "whole_datamap" : self.synapse.frame,
                "datamap_pixelsize" : self.microscope_generator.pixelsize
            }
        )

        super().reset(seed=seed)

        # Updates the current bleach function
        self.microscope = self.microscope_generator.generate_microscope(
            fluo_params=self.bleach_sampler.sample()
        )

        self.current_step = 0
        self.episode_memory = {
            "actions" : [],
            "mo_objs" : [],
            "reward" : [],
        }

        state = self._update_datamap()

        self.state = numpy.stack((state, numpy.zeros_like(state), numpy.zeros_like(state)), axis=-1)
        return (self.state.astype(numpy.uint16), numpy.zeros((self.observation_space[1].shape[0],), dtype=numpy.float32)), {}

    def _update_datamap(self):
        # Acquire confocal image which sets the current state
        conf_params = self.microscope_generator.generate_params()
        state, _, _ = self.microscope.get_signal_and_bleach(
            self.datamap, self.datamap.pixelsize, **conf_params, bleach=False
        )
        return state

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
        done = done or self.datamap["base"].max() == 0

        self.current_step += 1
        self.episode_memory["mo_objs"].append(mo_objs)
        self.episode_memory["actions"].append(action)
        self.episode_memory["reward"].append(reward)

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

        state = self._update_datamap()
        self.state = numpy.stack((state, conf1, sted_image), axis=-1)

        return (self.state.astype(numpy.uint16), obs.astype(numpy.float32)), reward, done, False, info
