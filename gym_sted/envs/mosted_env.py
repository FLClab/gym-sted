
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
                    max_episode_steps=10, scale_nanodomain_reward=1.,
                    normalize_observations=False):

        self.actions = actions
        self.action_space = spaces.Box(
            low=numpy.array([defaults.action_spaces[name]["low"] for name in self.actions], dtype=numpy.float32),
            high=numpy.array([defaults.action_spaces[name]["high"] for name in self.actions], dtype=numpy.float32),
            dtype=numpy.float32
        )
        self.observation_space = spaces.Tuple((
            spaces.Box(0, 2**16, shape=(64, 64, 3), dtype=numpy.uint16),
            spaces.Box(
                0, 1024, shape=(len(self.obj_names) * max_episode_steps + len(self.actions) * max_episode_steps,),
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

        molecules = 5
        self.synapse_generator = SynapseGenerator(
            mode="mushroom", seed=None, n_nanodomains=(3, 15), n_molecs_in_domain=(molecules * 20, molecules * 35)
        )
        self.microscope_generator = MicroscopeGenerator()
        self.microscope = self.microscope_generator.generate_microscope()
        if isinstance(self.bleach_sampling, dict):
            self.bleach_sampler = BleachSampler(**self.bleach_sampling)
        else:
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
        self.normalize_observations = normalize_observations
        self.action_normalizer = Normalizer(self.actions, defaults.action_spaces)
        self.obj_normalizer = Normalizer(self.obj_names, scales_dict)

    def step(self, action):
        """
        Method that should be implemented in the object that inherited

        :param action: A `numpy.ndarray` of the action
        """
        raise NotImplementedError

    def reset(self, seed=None, options=None):
        """
        Resets the environment with a new datamap

        :returns : A `numpy.ndarray` of the molecules
        """
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
