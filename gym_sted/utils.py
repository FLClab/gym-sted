
import numpy
import random
import warnings

from skimage import filters

from pysted import base, utils
from pysted import exp_data_gen as dg

from . import defaults

def get_foreground(img):
    """Return a background mask of the given image using the OTSU method to threshold.

    :param 2d-array img: The image.

    :returns: A mask (2d array of bool: True on foreground, False on background).
    """
    val = filters.threshold_otsu(img)
    return img > val

class SynapseGenerator():
    """
    Creates a synapse generator
    """
    def __init__(self, molecules=5, n_nanodomains=40, n_molecs_in_domain=25,
                    min_dist=100, valid_thickness=(3, 10), mode="rand", seed=None):
        # Assigns member variables
        self.molecules = molecules
        self.n_nanodomains = n_nanodomains
        self.n_molecs_in_domain = n_molecs_in_domain
        self.min_dist = min_dist
        self.valid_thickness = valid_thickness
        self.mode = mode
        self.seed = seed

    def __call__(self, *args, **kwargs):
        """
        Implements the `call` method of the class.

        :returns : A `numpy.ndarray` of the molecules
        """
        return self.generate(*args, **kwargs)

    def generate(self, rotate=False, *args, **kwargs):
        """
        Generates the molecule disposition

        :returns : A `numpy.ndarray` of the molecules
        """
        mode = kwargs.get("mode", None)
        if mode is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                synapse = dg.Synapse(self.molecules, mode=self.mode, seed=self.seed)
                synapse.add_nanodomains(
                    self.n_nanodomains, min_dist_nm=self.min_dist, seed=self.seed,
                    n_molecs_in_domain=self.n_molecs_in_domain, valid_thickness=self.valid_thickness
                )
                if rotate:
                    synapse.rotate_and_translate()
        else:
            seed = kwargs.get("seed", None)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                synapse = dg.Synapse(self.molecules, mode=self.mode, seed=seed)
                synapse.add_nanodomains(
                    self.n_nanodomains, min_dist_nm=self.min_dist, seed=seed,
                    n_molecs_in_domain=self.n_molecs_in_domain, valid_thickness=self.valid_thickness
                )
                if rotate:
                    synapse.rotate_and_translate()

        return synapse

class MoleculesGenerator():
    """
    Generate a datamap with randomly located molecules.

    :param shape: A tuple representing the shape of the datamap. If only 1 number is passed, a square datamap will be
                  generated.
    :param sources: Number of molecule sources to be randomly placed on the datamap.
    :param molecules: Average number of molecules contained on each source. The actual number of molecules will be
                      determined by poisson sampling.
    :param shape_sources : A `tuple` of the shape of the sources
    :param random_state: Sets the seed of the random number generator.
    """
    def __init__(self, shape, sources, molecules, shape_sources=(1, 1), random_state=None):
        # Instantiates the datamap generator
        self.shape = shape
        self.sources = sources
        self.molecules = molecules
        self.shape_sources = shape_sources
        self.random_state = random_state

    def __call__(self):
        """
        Implements the `call` method of the class.

        :returns : A `numpy.ndarray` containing the randomly placed molecules and positions
        """
        return self.generate()

    def generate(self):
        """
        Generates the datamap

        :returns: A datamap containing the randomly placed molecules
        """
        numpy.random.seed(self.random_state)
        if type(self.shape) == int:
            shape = (self.shape, self.shape)
        else:
            shape = self.shape
        datamap = numpy.zeros(shape)
        pos = []
        for i in range(self.sources):
            row, col = numpy.random.randint(0, shape[0] - self.shape_sources[0]), numpy.random.randint(0, shape[1] - self.shape_sources[1])
            datamap[row : row + self.shape_sources[0], col : col + self.shape_sources[1]] = numpy.random.poisson(self.molecules)
            pos.append([row + self.shape_sources[0] // 2, row + self.shape_sources[1] // 2])
        return datamap, numpy.array(pos)

class MicroscopeGenerator():
    """
    Generate a Microscope configuration
    """
    def __init__(self, **kwargs):

        # Creates a default datamap
        delta = 1
        num_mol = 2
        self.molecules_disposition = numpy.zeros((50, 50))
        for j in range(1,4):
            for i in range(1,4):
                self.molecules_disposition[
                    j * self.molecules_disposition.shape[0]//4 - delta : j * self.molecules_disposition.shape[0]//4 + delta + 1,
                    i * self.molecules_disposition.shape[1]//4 - delta : i * self.molecules_disposition.shape[1]//4 + delta + 1] = num_mol

        # Extracts params
        self.laser_ex_params = kwargs.get("laser_ex", defaults.LASER_EX)
        self.laser_sted_params = kwargs.get("laser_sted", defaults.LASER_STED)
        self.detector_params = kwargs.get("detector", defaults.DETECTOR)
        self.objective_params = kwargs.get("objective", defaults.OBJECTIVE)
        self.fluo_params = kwargs.get("fluo", defaults.FLUO)
        self.pixelsize = 20e-9

    def generate_microscope(self, **kwargs):
        """
        Generates the `microscope` object
        """
        # Generating objects necessary for acquisition simulation
        laser_ex = base.GaussianBeam(**self.laser_ex_params)
        laser_sted = base.DonutBeam(**self.laser_sted_params)
        detector = base.Detector(**self.detector_params)
        objective = base.Objective(**self.objective_params)
        if "phy_react" in kwargs:
            tmp = self.fluo_params.copy()
            tmp["phy_react"] = kwargs.get("phy_react")
            fluo = base.Fluorescence(**tmp)
        else:
            fluo = base.Fluorescence(**self.fluo_params)

        self.microscope = base.Microscope(laser_ex, laser_sted, detector, objective, fluo)
        i_ex, _, _ = self.microscope.cache(self.pixelsize, save_cache=True)

        return self.microscope

    def generate_datamap(self, **kwargs):
        """
        Generates the `datamap` object
        """
        datamap_params = kwargs.get("datamap", {
            "whole_datamap" : self.molecules_disposition,
            "datamap_pixelsize" : self.pixelsize
        })

        i_ex, _, _ = self.microscope.cache(self.pixelsize, save_cache=True)
        datamap = base.Datamap(**datamap_params)
        datamap.set_roi(i_ex, "max")

        return datamap

    def generate_params(self, **kwargs):
        """
        Generates the confocal parameters
        """
        imaging_params = kwargs.get("imaging", {
            "pdt" : defaults.PDT,
            "p_ex" : defaults.P_EX,
            "p_sted" : 0.
        })
        return imaging_params

class BleachSampler:
    """
    Creates a `BleachSampler` to sample new bleaching function parameters

    :param mode: The sampling mode from {constant, uniform, choice}
    """
    def __init__(self, mode, value=None, seed=None):
        self.seed(seed)
        self.mode = mode
        self.value = value
        self.uniform_limits = [
            (0.001e-5, 0.015e-5), # p_ex
            (0.004e-8, 0.012e-8) # p_sted
        ]
        self.normal_limits = [
            (0.008e-5, 0.007e-5 / 2.576), # p_ex
            (0.008e-8, 0.004e-8 / 2.576) # p_sted
        ]
        self.choices = [
            (0.008e-5 - 0.007e-5, 0.008e-5, 0.008e-5 + 0.007e-5), # p_ex
            (0.008e-8 - 0.004e-8, 0.008e-8, 0.008e-8 + 0.004e-8) # p_sted
        ]
        self.sampling_method = getattr(self, "_{}_sample".format(self.mode))

    def seed(self, seed=None):
        """
        Seeds the sampler
        """
        numpy.random.seed(seed)
        random.seed(seed)

    def sample(self):
        """
        Implements the sample method

        :returns : A `dict` of `phy_react`
        """
        return self.sampling_method()

    def _constant_sample(self):
        """
        Implements a constant sampling of the bleach parameters
        """
        if self.value:
            return self.value
        return defaults.FLUO["phy_react"]

    def _uniform_sample(self):
        """
        Implements a uniform sampling of the bleach parameters
        """
        tmp = defaults.FLUO["phy_react"].copy()
        for key, (m, M) in zip(tmp.keys(), self.uniform_limits):
            tmp[key] = random.uniform(m, M)
        return tmp

    def _normal_sample(self):
        """
        Implements a normal sampling of the bleach parameters
        """
        tmp = defaults.FLUO["phy_react"].copy()
        for key, (mu, std) in zip(tmp.keys(), self.normal_limits):
            tmp[key] = random.gauss(mu, std)
            if tmp[key] < 0:
                while tmp[key] < 0:
                    tmp[key] = random.gauss(mu, std)
        return tmp

    def _choice_sample(self):
        """
        Implements a choice sampling of the bleach parameters
        """
        tmp = defaults.FLUO["phy_react"].copy()
        for key, choices in zip(tmp.keys(), self.choices):
            tmp[key] = random.choice(choices)
        return tmp

class Normalizer:
    """
    Implements a `Normalizer`
    """
    def __init__(self, names, scales):
        """
        Instantiates the `Normalizer`

        :param names: A `list` of `str`
        :param scales: A `dict` of scales where each keys contain {'low', 'high'}
        """
        self.names = names
        self.scales = scales

    def __call__(self, x):
        """
        Implements the `__call__` method of the `Normalizer`
        """
        return self.normalize(x)

    def normalize(self, x):
        """
        Implements the normalize method of the class
        """
        if isinstance(x, (list, tuple)):
            x = numpy.array(x)
        return numpy.array([(_x - self.scales[name]["low"]) / (self.scales[name]["high"] - self.scales[name]["low"]) for name, _x in zip(self.names, x)])
