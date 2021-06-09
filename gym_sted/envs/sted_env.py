
import gym
import numpy
import random

from gym import error, spaces, utils
from gym.utils import seeding

from gym_sted.utils import DatamapGenerator, MicroscopeGenerator

from pysted.utils import Experiment
from banditopt.utils import get_foreground

class STEDEnv(gym.Env):
    """
    Creates a `STEDEnv`

    The `STEDEnv` implements a scan of the entire field of view
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):

        self.datamap_generator = DatamapGenerator(
            shape = (50, 50),
            sources = 10,
            molecules = 3,
            shape_sources = (2, 2)
        )
        self.microscope_generator = MicroscopeGenerator()
        self.microscope = self.microscope_generator.generate_microscope()

        self.action_space = spaces.Box(low=5e-6, high=5e-3, shape=(1,))

        self.viewer = None
        self.state = None
        self.reward_calculator = None

        self.seed()

    def step(self, action):

        # Generates the datamap and imaging parameters
        datamap, sted_params = self.microscope_generator.generate_datamap(
            datamap = {
                "whole_datamap" : self.state,
                "datamap_pixelsize" : self.microscope_generator.pixelsize
            },
            imaging = {
                "pdt" : 100.0e-6,
                "p_ex" : 2.0e-6,
                "p_sted" : action[0]
            }
        )
        _, conf_params = self.microscope_generator.generate_datamap()

        # Creates the Experiment
        # By using the same datamap we ensure that it is shared
        experiment = Experiment()
        experiment.add("STED", self.microscope, datamap, sted_params)
        experiment.add("conf", self.microscope, datamap, conf_params)

        # Acquire confocal image
        name, history = experiment.acquire("conf", 1, bleach=False)
        conf1 = history["acquisition"][-1]

        # Acquire DyMIN image
        name, history = experiment.acquire("STED", 1, bleach=True)
        sted_image = history["acquisition"][-1]

        # Acquire confocal image
        name, history = experiment.acquire("conf", 1, bleach=False)
        conf2 = history["acquisition"][-1]

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
        info = {}
        observation = None
        return observation, reward, done, info

    def reset(self):
        """
        Resets the environment with a new datamap

        :returns : A `numpy.ndarray` of the molecules
        """
        self.state, positions = self.datamap_generator.generate()
        return self.state

    def render(self, mode='human'):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def update_(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def close(self):
        return None
