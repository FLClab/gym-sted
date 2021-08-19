
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
                    min_dist=100, valid_thickness=3, mode="rand", seed=None):
        # Assigns member variables
        self.molecules = molecules
        self.n_nanodomains = n_nanodomains
        self.n_molecs_in_domain = n_molecs_in_domain
        self.min_dist = min_dist
        self.valid_thickness = valid_thickness
        self.mode = mode
        self.seed = seed

    def __call__(self):
        """
        Implements the `call` method of the class.

        :returns : A `numpy.ndarray` of the molecules
        """
        return self.generate()

    def generate(self):
        """
        Generates the molecule disposition

        :returns : A `numpy.ndarray` of the molecules
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            synapse = dg.Synapse(self.molecules, mode=self.mode, seed=self.seed)
            synapse.add_nanodomains(
                self.n_nanodomains, min_dist_nm=self.min_dist, seed=self.seed,
                n_molecs_in_domain=self.n_molecs_in_domain, valid_thickness=self.valid_thickness
            )
        return synapse.frame

class SynapseGenerator2():
    """
    Creates a synapse generator
    """
    def __init__(self, molecules=5, n_nanodomains=40, n_molecs_in_domain=25,
                    min_dist=(50, 100), valid_thickness=(3, 10), mode="rand", seed=None):
        # Assigns member variables
        self.molecules = molecules
        self.n_nanodomains = n_nanodomains
        self.n_molecs_in_domain = n_molecs_in_domain
        self.min_dist = min_dist
        self.valid_thickness = valid_thickness
        self.mode = mode
        self.seed = seed

    def __call__(self):
        """
        Implements the `call` method of the class.

        :returns : A `numpy.ndarray` of the molecules
        """
        return self.generate()

    def generate(self):
        """
        Generates the molecule disposition

        :returns : A `numpy.ndarray` of the molecules
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            synapse = dg.Synapse(self.molecules, mode=self.mode, seed=self.seed)
            synapse.add_nanodomains(
                self.n_nanodomains, min_dist_nm=self.min_dist, seed=self.seed,
                n_molecs_in_domain=self.n_molecs_in_domain, valid_thickness=self.valid_thickness
            )
        # return synapse.frame
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

        datamap_params = kwargs.get("datamap", {
            "whole_datamap" : self.molecules_disposition,
            "datamap_pixelsize" : self.pixelsize
        })

        i_ex, _, _ = self.microscope.cache(self.pixelsize, save_cache=True)
        datamap = base.Datamap(**datamap_params)
        datamap.set_roi(i_ex, "max")

        return datamap

    def generate_temporal_datamap(self, **kwargs):
        # jveux tu mettre les params pour créer le tstack ici ou jveux gérer ça direct dans l'env?
        temporal_datamap_params = kwargs.get("temporal_datamap", {
            "whole_datamap": kwargs.get("whole_datamap", self.molecules_disposition),
            "datamap_pixelsize": kwargs.get("datamap_pixelsize", self.pixelsize),
            "synapse_obj": kwargs.get("synapse_obj", None)
        })

        decay_time_us = kwargs.get("decay_time_us", 1000000)
        n_decay_steps = kwargs.get("n_decay_steps", 10)
        # print(decay_time_us)
        # for now I will create a TestTemporalDmap obj, but eventually this should be a TemporalSynapseDmap obj
        i_ex, _, _ = self.microscope.cache(self.pixelsize, save_cache=True)
        # temporal_datamap = base.TestTemporalDmap(**temporal_datamap_params)
        temporal_datamap = base.TemporalSynapseDmap(**temporal_datamap_params)
        temporal_datamap.set_roi(i_ex, "max")
        temporal_datamap.create_t_stack_dmap(decay_time_us, n_decay_steps=n_decay_steps)
        temporal_datamap.update_whole_datamap(0)

        return temporal_datamap

    def generate_temporal_datamap_smoother_flash(self, **kwargs):
        # jveux tu mettre les params pour créer le tstack ici ou jveux gérer ça direct dans l'env?
        temporal_datamap_params = kwargs.get("temporal_datamap", {
            "whole_datamap": kwargs.get("whole_datamap", self.molecules_disposition),
            "datamap_pixelsize": kwargs.get("datamap_pixelsize", self.pixelsize),
            "synapse_obj": kwargs.get("synapse_obj", None),
        })

        decay_time_us = kwargs.get("decay_time_us", 1000000)
        n_decay_steps = kwargs.get("n_decay_steps", 10)
        flash_delay = kwargs.get("flash_delay", 2)
        # print(decay_time_us)
        # for now I will create a TestTemporalDmap obj, but eventually this should be a TemporalSynapseDmap obj
        i_ex, _, _ = self.microscope.cache(self.pixelsize, save_cache=True)
        # temporal_datamap = base.TestTemporalDmap(**temporal_datamap_params)
        temporal_datamap = base.TemporalSynapseDmap(**temporal_datamap_params)
        temporal_datamap.set_roi(i_ex, "max")
        temporal_datamap.create_t_stack_dmap_smooth(decay_time_us, n_decay_steps=n_decay_steps, delay=flash_delay)
        temporal_datamap.update_whole_datamap(0)

        return temporal_datamap

    def generate_temporal_datamap_sampled_flash(self, **kwargs):
        temporal_datamap_params = kwargs.get("temporal_datamap", {
            "whole_datamap": kwargs.get("whole_datamap", self.molecules_disposition),
            "datamap_pixelsize": kwargs.get("datamap_pixelsize", self.pixelsize),
            "synapse_obj": kwargs.get("synapse_obj", None),
        })

        decay_time_us = kwargs.get("decay_time_us", 1000000)
        n_decay_steps = kwargs.get("n_decay_steps", 10)
        flash_delay = kwargs.get("flash_delay", 2)
        # print(decay_time_us)
        # for now I will create a TestTemporalDmap obj, but eventually this should be a TemporalSynapseDmap obj
        i_ex, _, _ = self.microscope.cache(self.pixelsize, save_cache=True)
        # temporal_datamap = base.TestTemporalDmap(**temporal_datamap_params)
        temporal_datamap = base.TemporalSynapseDmap(**temporal_datamap_params)
        temporal_datamap.set_roi(i_ex, "max")
        temporal_datamap.create_t_stack_dmap_sampled(decay_time_us, n_decay_steps=n_decay_steps, delay=flash_delay,
                                                     curves_path="audurand_pysted/flash_files/events_curves.npy")
        temporal_datamap.update_whole_datamap(0)

        return temporal_datamap

    def generate_params(self, **kwargs):

        imaging_params = kwargs.get("imaging", {
            "pdt" : 100e-6,
            "p_ex" : 2e-6,
            "p_sted" : 0.
        })
        return imaging_params

class BleachSampler:
    """
    Creates a `BleachSampler` to sample new bleaching function parameters

    :param mode: The sampling mode from {constant, uniform, choice}
    """
    def __init__(self, mode):

        self.mode = mode
        self.uniform_limits = [
            (0.25e-8, 0.25e-6), # p_ex
            (100.0e-11, 15.0e-11) # p_sted
        ]
        self.normal_limits = [
            (0.25e-7, 1.e-7), # p_ex
            (25.0e-11, 100e-11) # p_sted
        ]
        self.choices = [
            (0.01e-7, 0.25e-7, 1.0e-7), # p_ex
            (15.0e-11, 25.0e-11, 100.0e-11) # p_sted
        ]
        self.sampling_method = getattr(self, "_{}_sample".format(self.mode))

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
        return tmp

    def _choice_sample(self):
        """
        Implements a choice sampling of the bleach parameters
        """
        tmp = defaults.FLUO["phy_react"].copy()
        for key, choices in zip(tmp.keys(), self.choices):
            tmp[key] = random.choice(choices)
        return tmp

class RecordingQueue:
    def __init__(self, object: object, maxlen: int, num_sensors: tuple):
        self.rec_queue: numpy.array = numpy.zeros(shape=(maxlen, *num_sensors), dtype=numpy.int32) #allocate the memory we need ahead of time
        self.max_length: int = maxlen
        self.queue_tail: int = maxlen - 1
        if (len(object) > 0):
            for val in object:
                self.enqueue(val)

    def to_array(self) -> numpy.array:
        head = (self.queue_tail + 1) % self.max_length
        return numpy.roll(self.rec_queue, -head, axis=0) # this will force a copy

    def enqueue(self, new_data: numpy.array) -> None:
        # move tail pointer forward then insert at the tail of the queue
        # to enforce max length of recording
        self.queue_tail = (self.queue_tail + 1) % self.max_length
        self.rec_queue[self.queue_tail] = new_data

    def peek(self) -> int:
        queue_head = (self.queue_tail + 1) % self.max_length
        return self.rec_queue[queue_head]
    def item_at(self, index: int) -> int:
        # the item we want will be at head + index
        loc = (self.queue_tail + 1 + index) % self.max_length
        return self.rec_queue[loc]
    def replace_item_at(self, index: int, newItem: int):
        # the item we want will be at head + index
        loc = (self.queue_tail + 1 + index) % self.max_length
        self.rec_queue[loc] = newItem
    def __repr__(self):
        return "tail: " + str(self.queue_tail) + "\narray: " + str(self.rec_queue)
    def __str__(self):
        return "tail: " + str(self.queue_tail) + "\narray:\n" + str(self.rec_queue)
        return str(self.to_array())
