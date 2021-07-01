
import gym
import numpy
import random

from gym import error, spaces, utils
from gym.utils import seeding
from matplotlib import pyplot

from gym_sted.utils import SynapseGenerator, MicroscopeGenerator, get_foreground


class STEDEnv(gym.Env):
    """
    Creates a `STEDEnv`

    The `STEDEnv` implements a scan of the entire field of view
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):

        self.synapse_generator = SynapseGenerator()
        self.microscope_generator = MicroscopeGenerator()
        self.microscope = self.microscope_generator.generate_microscope()

        self.action_space = spaces.Box(low=5e-6, high=5e-3, shape=(1,))

        self.state = None
        self.initial_count = None
        self.reward_calculator = None
        self.datamap = None
        self.viewer = None

        self.seed()

    def step(self, action):

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

        reward = self.reward_calculator.evaluate(sted_image, conf1, conf2, fg_s, fg_c)

        done = True
        observation = conf2
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
        return self.state

    def render(self, info, mode='human'):
        """
        Renders the environment

        :param info: A `dict` of data
        :param mode: A `str` of the available mode
        """
        fig, axes = pyplot.subplots(1, 3, figsize=(10,3), sharey=True, sharex=True)

        axes[0].imshow(info["conf1"])
        axes[0].set_title(f"Datamap roi")

        axes[1].imshow(info["bleached"]["base"][self.datamap.roi])
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

    def close(self):
        return None
