
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
from gym_sted.datamaps import DatamapGenerator

from .mosted_env import STEDMultiObjectivesEnv

class PreferenceSTEDMultiObjectivesEnv(STEDMultiObjectivesEnv):
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
                    max_episode_steps=30, scale_nanodomain_reward=1.,
                    normalize_observations=True, max_count_rate=20e+6,
                    **kwargs):

        self.group = None
        self.max_count_rate = max_count_rate
        self.conf_params = None

        super(PreferenceSTEDMultiObjectivesEnv, self).__init__(
            bleach_sampling = bleach_sampling,
            actions = actions,
            max_episode_steps = max_episode_steps,
            scale_nanodomain_reward = scale_nanodomain_reward,
            normalize_observations=normalize_observations,
            **kwargs
        )
        
        # Update observation space
        self.observation_space = spaces.Tuple((
            spaces.Box(0, 2**16, shape=(96, 96, 3), dtype=numpy.uint16),
            spaces.Box(
                0, 1024, shape=(len(self.obj_names) * max_episode_steps + len(self.actions) * max_episode_steps,),
                dtype=numpy.float32
            ) # Articulation, shape is given by objectives, actions at each steps
        ))        

        self.datamap_generator = DatamapGenerator(
            molecules=40, molecules_scale=0.1, shape=(96, 96),
            path=defaults.DATAMAP_PATH, augment=True
        )

    def step(self, action):

        # Action is an array of size self.actions and main_action
        # main action should be in the [0, 1, 2]
        # We manually clip the actions which are out of action space
        action = numpy.clip(action, self.action_space.low, self.action_space.high)

        # Acquire an image with the given parameters
        sted_image, bleached, conf1, conf2, fg_s, fg_c = self._acquire(action)
        mo_objs = self.mo_reward_calculator.evaluate(sted_image, conf1, conf2, fg_s, fg_c)

        # Reward is given by the objectives
        reward, _, _ = self.preference_articulation.articulate(
            [mo_objs]
        )
        reward = reward.item()

        # print("STED: {:0.2f} - Exc: {:0.2f} - Pdt: {:0.2f}".format(
        #     action[0] * 1e+3, action[1] * 1e+6, action[2] * 1e+6
        # ))
        # print("Res: {:0.2f} - Pb: {:0.2f} - SNR: {:0.2f}".format(
        #     *mo_objs
        # ))
        # print(reward)     
        # print()

        # Updates memory
        done = self.current_step >= self.spec.max_episode_steps - 1
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
            "sample" : self.group
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

        self.conf_params = None

        # Samples the current samply type 
        self.group = self.datamap_generator.sample_group()
        self.bleach_sampler.optimizer.set_correction_factor(self.group)

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
        datamap = self.datamap_generator(group=self.group)
        self.datamap = self.microscope_generator.generate_datamap(
            datamap = {
                "whole_datamap" : datamap,
                "datamap_pixelsize" : self.microscope_generator.pixelsize
            }
        )

        # Acquire confocal image which sets the current state and defines the confocal 
        # parameters to avoid the saturation of detector
        if isinstance(self.conf_params, type(None)):
            conf_params = self.microscope_generator.generate_params()
            state, _, _ = self.microscope.get_signal_and_bleach(
                self.datamap, self.datamap.pixelsize, **conf_params, bleach=False
            )
            while state.max() / conf_params["pdt"] > self.max_count_rate:
                conf_params["p_ex"] *= 0.75
                state, _, _ = self.microscope.get_signal_and_bleach(
                    self.datamap, self.datamap.pixelsize, **conf_params, bleach=False
                )
            self.conf_params = conf_params
        state, _, _ = self.microscope.get_signal_and_bleach(
            self.datamap, self.datamap.pixelsize, **self.conf_params, bleach=False
        )
        return state
    
class PreferenceCountRateScaleSTEDMultiObjectivesEnv(PreferenceSTEDMultiObjectivesEnv):
    """
    Creates a `PreferenceCountRateScaleSTEDMultiObjectivesEnv`. In this case the 
    reward is still calculated using the `PrefNet` model with a sigmoid activation 
    to normalize the scores within [0, 1]. We give negative feedback when the model 
    selects actions that leads to >1MHz photons in the acquired STED image

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
                    max_episode_steps=30, scale_nanodomain_reward=1.,
                    normalize_observations=True, max_count_rate=20e+6,
                    negative_reward=-10., **kwargs):

        super(PreferenceCountRateScaleSTEDMultiObjectivesEnv, self).__init__(
            bleach_sampling = bleach_sampling,
            actions = actions,
            max_episode_steps = max_episode_steps,
            scale_nanodomain_reward = scale_nanodomain_reward,
            normalize_observations=normalize_observations,
            max_count_rate=max_count_rate,
            **kwargs
        )

        self.negative_reward = negative_reward

        # Update observation space
        self.observation_space = spaces.Tuple((
            spaces.Box(0, 2**16, shape=(96, 96, 3), dtype=numpy.uint16),
            spaces.Box(
                0, 1, shape=(1 + len(self.obj_names) * max_episode_steps + len(self.actions) * max_episode_steps,),
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

        # Reward is given by the objectives
        reward, _, _ = self.preference_articulation.articulate(
            [mo_objs], use_sigmoid=False
        )
        reward = reward.item()

        # Reward will be dependant on the count rate of the STED image
        # We extract the pixel dwelltime used to acquire
        sted_params = self.microscope_generator.generate_params(
            imaging = {
                name : action[self.actions.index(name)]
                    if name in self.actions else getattr(defaults, name.upper())
                    for name in ["pdt", "p_ex", "p_sted"]
            }
        )
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
            "bleached" : bleached,
            "sted_image" : sted_image,
            "conf1" : conf1,
            "conf2" : conf2,
            "fg_c" : fg_c,
            "fg_s" : fg_s,
            "mo_objs" : mo_objs,
            "reward" : reward,
            "sample" : self.group,
            "conf_params" : self.conf_params
        }

        conf_params = self.microscope_generator.generate_params()
        # Build the observation space
        obs = [self.conf_params["p_ex"] / conf_params["p_ex"]]
        for a, mo in zip(self.episode_memory["actions"], self.episode_memory["mo_objs"]):
            obs.extend(self.action_normalizer(a) if self.normalize_observations else a)
            obs.extend(self.obj_normalizer(mo) if self.normalize_observations else mo)
        obs = numpy.pad(numpy.array(obs), (0, self.observation_space[1].shape[0] - len(obs)))
        state = self._update_datamap()
        self.state = numpy.stack((state, conf1, sted_image), axis=-1)

        return (self.state.astype(numpy.uint16), obs.astype(numpy.float32)), reward, done, False, info        
    
class RecurrentPreferenceCountRateScaleSTEDMultiObjectivesEnv(PreferenceSTEDMultiObjectivesEnv):
    """
    Creates a `PreferenceCountRateScaleSTEDMultiObjectivesEnv`. In this case the 
    reward is still calculated using the `PrefNet` model with a sigmoid activation 
    to normalize the scores within [0, 1]. We give negative feedback when the model 
    selects actions that leads to >1MHz photons in the acquired STED image

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
                    max_episode_steps=30, scale_nanodomain_reward=1.,
                    normalize_observations=True, max_count_rate=20e+6,
                    negative_reward=-10.):

        super(RecurrentPreferenceCountRateScaleSTEDMultiObjectivesEnv, self).__init__(
            bleach_sampling = bleach_sampling,
            actions = actions,
            max_episode_steps = max_episode_steps,
            scale_nanodomain_reward = scale_nanodomain_reward,
            normalize_observations=normalize_observations,
            max_count_rate=max_count_rate
        )

        self.negative_reward = negative_reward

        # Update observation space
        self.observation_space = spaces.Tuple((
            spaces.Box(0, 2**16, shape=(96, 96, 3), dtype=numpy.uint16),
            spaces.Box(
                0, 1, shape=(1 + 1 + len(self.obj_names) + len(self.actions),),
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

        # Reward is given by the objectives
        reward, _, _ = self.preference_articulation.articulate(
            [mo_objs], use_sigmoid=False
        )
        reward = reward.item()

        # Reward will be dependant on the count rate of the STED image
        # We extract the pixel dwelltime used to acquire
        sted_params = self.microscope_generator.generate_params(
            imaging = {
                name : action[self.actions.index(name)]
                    if name in self.actions else getattr(defaults, name.upper())
                    for name in ["pdt", "p_ex", "p_sted"]
            }
        )
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
            "bleached" : bleached,
            "sted_image" : sted_image,
            "conf1" : conf1,
            "conf2" : conf2,
            "fg_c" : fg_c,
            "fg_s" : fg_s,
            "mo_objs" : mo_objs,
            "reward" : reward,
            "sample" : self.group,
            "conf_params" : self.conf_params
        }

        conf_params = self.microscope_generator.generate_params()
        # Build the observation space
        obs = [
            self.conf_params["p_ex"] / conf_params["p_ex"], 
            count_rate / self.max_count_rate,
            *self.action_normalizer(self.episode_memory["actions"][-1]),
            *self.obj_normalizer(self.episode_memory["mo_objs"][-1])
        ]
        
        state = self._update_datamap()
        self.state = numpy.stack((state, conf1, sted_image), axis=-1)

        return (self.state.astype(numpy.uint16), obs.astype(numpy.float32)), reward, done, False, info            