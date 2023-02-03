
import numpy
import random
import warnings

from skimage import filters
from scipy import optimize

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

        # Generating objects necessary for acquisition simulation
        laser_ex = base.GaussianBeam(**kwargs.get("laser_ex_params", self.laser_ex_params))
        laser_sted = base.DonutBeam(**kwargs.get("laser_sted_params", self.laser_sted_params))
        detector = base.Detector(**kwargs.get("detector_params", self.detector_params))
        objective = base.Objective(**kwargs.get("objective_params", self.objective_params))
        fluo = base.Fluorescence(**kwargs.get("fluo_params", self.fluo_params))

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
        individual_flashes = kwargs.get("individual_flashes", False)
        # print(decay_time_us)
        # for now I will create a TestTemporalDmap obj, but eventually this should be a TemporalSynapseDmap obj
        i_ex, _, _ = self.microscope.cache(self.pixelsize, save_cache=True)
        # temporal_datamap = base.TestTemporalDmap(**temporal_datamap_params)
        temporal_datamap = base.TemporalSynapseDmap(**temporal_datamap_params)
        temporal_datamap.set_roi(i_ex, "max")
        temporal_datamap.create_t_stack_dmap_smooth(decay_time_us, n_decay_steps=n_decay_steps, delay=flash_delay,
                                                    individual_flashes=individual_flashes)
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
        individual_flashes = kwargs.get("individual_flashes", False)
        # print(decay_time_us)
        # for now I will create a TestTemporalDmap obj, but eventually this should be a TemporalSynapseDmap obj
        i_ex, _, _ = self.microscope.cache(self.pixelsize, save_cache=True)
        # temporal_datamap = base.TestTemporalDmap(**temporal_datamap_params)
        temporal_datamap = base.TemporalSynapseDmap(**temporal_datamap_params)
        temporal_datamap.set_roi(i_ex, "max")
        temporal_datamap.create_t_stack_dmap_sampled(decay_time_us, n_decay_steps=n_decay_steps, delay=flash_delay,
                                                     curves_path="audurand_pysted/flash_files/events_curves.npy",
                                                     individual_flashes=individual_flashes)
        temporal_datamap.update_whole_datamap(0)

        return temporal_datamap

    def generate_params(self, **kwargs):

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
    def __init__(self, mode, value=None, seed=None, criterions=None):
        self.seed(seed)
        self.mode = mode
        self.value = value

        if isinstance(criterions, type(None)):
            self.criterions = defaults.fluorescence_criterions
        else:
            self.criterions = criterions
        self.optimizer = FluorescenceOptimizer()

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

        :returns : A `dict` of fluorescent parameters
        """
        return self.sampling_method()

    def _constant_sample(self):
        """
        Implements a constant sampling of the bleach parameters
        """
        if self.value:
            return self.value
        return defaults.FLUO

    def _uniform_sample(self):
        """
        Implements a uniform sampling of the bleach parameters
        """
        fluo = defaults.FLUO.copy()

        criterion = UniformCriterion(self.criterions)
        optimized = self.optimizer.optimize(criterion)

        for objective, params in optimized.items():
            for key, value in params.items():
                fluo[key] = value
        return fluo

    # def _normal_sample(self):
    #     """
    #     Implements a normal sampling of the bleach parameters
    #     """
    #     fluo = defaults.FLUO.copy()
    #     for key, (mu, std) in self.normal_limits.items():
    #         fluo[key] = random.gauss(mu, std)
    #         while fluo[key] < 0:
    #             fluo[key] = random.gauss(mu, std)
    #     return fluo

    def _choice_sample(self):
        """
        Implements a choice sampling of the bleach parameters
        """
        fluo = defaults.FLUO.copy()

        criterion = ChoiceCriterion(self.criterions)
        optimized = self.optimizer.optimize(criterion)

        for objective, params in optimized.items():
            for key, value in params.items():
                fluo[key] = value
        return fluo

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

class FluorescenceOptimizer():
    """
    Optimizes the parameters of fluorescence to obtain the
    given photobleaching and signal in the acquired images.
    """
    FACTORS = {
        "clusters" : 2.25, # CaMKII & PSD95
        "actin" : 3.0,
        "tubulin" : 3.75
    }

    def __init__(self, microscope=None, sample="clusters", iterations=10, pixelsize=20e-9):
        """
        Instantiates the `FluorescenceOptimizer`

        :param microscope: A `pysted.base.Microscope` object
        :param sample: A `str` of the sample type that is being optimized
        :param iterations: An `int` of the number of iterations to perform
        """
        self.microscope = microscope
        if isinstance(self.microscope, type(None)):
            self.microscope = MicroscopeGenerator().generate_microscope()
        self.iterations = iterations
        self.pixelsize = pixelsize

        assert sample in ["clusters", "actin", "tubulin"]
        self.correction_factor = self.FACTORS[sample]
        self.scale_factor = 40.

        # This seems to be optimal for a starting point
        self.microscope.fluo.sigma_abs = defaults.FLUO["sigma_abs"]
        self.microscope.fluo.k1 = defaults.FLUO["k1"]
        self.microscope.fluo.b = defaults.FLUO["b"]

    def default_parameters(self):
        """
        Returns the default parameters
        """
        self.microscope.fluo.sigma_abs = defaults.FLUO["sigma_abs"]
        self.microscope.fluo.k1 = defaults.FLUO["k1"]
        self.microscope.fluo.b = defaults.FLUO["b"]
        return defaults.FLUO["k1"], defaults.FLUO["b"], defaults.FLUO["sigma_abs"]

    def aggregate(self, criterions, **kwargs):
        """
        Aggregates the returned values
        """
        out = {}
        if "bleach" in criterions:
            out["bleach"] = {
                "k1" : kwargs.get("k1") * 1e-15,
                "b" : kwargs.get("b")
            }
        if "signal" in criterions:
            sigma_abs = self.microscope.fluo.sigma_abs
            out["signal"] = {
                "sigma_abs" : {
                    int(self.microscope.excitation.lambda_ * 1e9) : kwargs.get("sigma_abs") * 1e-20,
                    int(self.microscope.sted.lambda_ * 1e9) : sigma_abs[int(self.microscope.sted.lambda_ * 1e9)]
                }
            }
        return out

    def optimize(self, criterions):
        """
        Optimizes the fluorescene parameters given the input criterions
        that have to be met. The optimization of the parameters is done
        sequentially has this seems to produce decent results on the datamaps
        that were tested. However, a multi-objective approach (e.g. NSGA-II)
        could be better suited to find the Pareto choices.

        :param criterions: A `dict` of criterions

        :returns : A `dict` of the optimized parameters

        :example :

        criterions = {
            "bleach" : {
                "p_ex" : <VALUE>,
                "p_sted" : <VALUE>,
                "pdt" : <VALUE>,
                "target" : <VALUE>
            },
            "signal" : {
                "p_ex" : <VALUE>,
                "p_sted" : <VALUE>,
                "pdt" : <VALUE>,
                "target" : <VALUE>
            },
        }
        params = optimizer.optimize(criterions)
        >>> params
        {
            "bleach" : {
                "k1" : <VALUE>,
                "b" : <VALUE>
            },
            "signal" : {
                "sigma_abs" : {
                    635 : <VALUE>,
                    750 : <VALUE>
                }
            }
        }
        """
        k1, b, sigma_abs = self.default_parameters()
        sigma_abs = sigma_abs[int(self.microscope.excitation.lambda_ * 1e9)]
        for _ in range(self.iterations):
            params = criterions.get("bleach", None)
            if params:
                # Optimize bleaching constants
                res = optimize.minimize(
                    self.optimize_bleach, x0=[k1, b],
                    args=(params["p_ex"], params["p_sted"], params["pdt"], params["target"]),
                    options={"eps" : 0.01, "maxiter": 100}, tol=1e-3,
                    bounds = [(0., numpy.inf), (0., 5.0)]
                )
                k1, b = res.x

            # Optimize signal constant
            params = criterions.get("signal", None)
            if params:
                res = optimize.minimize(
                    self.optimize_sigma_abs, x0=[sigma_abs],
                    args=(params["p_ex"], params["p_sted"], params["pdt"], params["target"]),
                    options={"eps" : 0.01, "maxiter": 100}, tol=1e-3,
                    bounds = [(0., numpy.inf)]
                )
                sigma_abs = res.x.item()
        return self.aggregate(criterions, k1=k1, b=b, sigma_abs=sigma_abs)

    def kb_map_to_im_bleach(self, kb_map, dwelltime, linestep):
        """
        Bleaching estimate for an infinite number of fluorophores
        kb_map being the map of k_bleach convolving each pixel
        """
        return 1 - numpy.exp((-kb_map * dwelltime * linestep).sum())

    def expected_bleach(self, p_ex, p_sted, pdt):
        """
        Calculates the expected confocal signal given some parameters

        :param p_ex: A `float` of the excitation power
        :param p_sted: A `float` of the STED power
        :param pdt: A `float` of the pixel dwelltime

        :returns : A `int` of the expected number of photons
        """
        __i_ex, __i_sted, psf_det = self.microscope.cache(self.pixelsize)

        i_ex = __i_ex * p_ex #the time averaged excitation intensity
        i_sted = __i_sted * p_sted #the instant sted intensity (instant p_sted = p_sted/(self.sted.tau * self.sted.rate))

        lambda_ex, lambda_sted = self.microscope.excitation.lambda_, self.microscope.sted.lambda_
        tau_sted = self.microscope.sted.tau
        tau_rep = 1 / self.microscope.sted.rate
        phi_ex =  self.microscope.fluo.get_photons(i_ex, lambda_=lambda_ex)
        phi_sted = self.microscope.fluo.get_photons(i_sted, lambda_=lambda_sted)

        kb_map = self.microscope.fluo.get_k_bleach(
            lambda_ex, lambda_sted, phi_ex, phi_sted*tau_sted/tau_rep, tau_sted,
            tau_rep, pdt
        )
        bleach = self.kb_map_to_im_bleach(kb_map, pdt, 1)
        return bleach

    def expected_confocal_signal(self, p_ex, p_sted, pdt):
        """
        Calculates the expected confocal signal given some parameters

        :param p_ex: A `float` of the excitation power
        :param p_sted: A `float` of the STED power
        :param pdt: A `float` of the pixel dwelltime

        :returns : A `int` of the expected number of photons
        """
        photons_mean = []
        # The calculation is repeated since there is randomness
        for _ in range(25):
            effective = self.microscope.get_effective(self.pixelsize, p_ex, p_sted)
            datamap = numpy.zeros_like(effective)
            cy, cx = (s // 2 for s in datamap.shape)
            datamap[cy, cx] = 1
            datamap = filters.gaussian(datamap, sigma=self.correction_factor)
            datamap = datamap / datamap.max() * self.scale_factor

            intensity = numpy.sum(effective * datamap)
            photons = self.microscope.fluo.get_photons(intensity)
            photons = self.microscope.detector.get_signal(photons, pdt, self.microscope.sted.rate)
            photons_mean.append(photons)
        photons = numpy.mean(photons_mean)
        return photons

    def optimize_bleach(self, x, p_ex, p_sted, pdt, target):
        """
        Method used by `scipy.optimize.minimize` to optimize the
        photobleaching
        """
        k1, b = x
        self.microscope.fluo.k1 = k1 * 1e-15
        self.microscope.fluo.b = b
        bleach = self.expected_bleach(p_ex, p_sted, pdt)
        error = (target - bleach) ** 2
        return error

    def optimize_sigma_abs(self, sigma_abs, p_ex, p_sted, pdt, target):
        """
        Method used by `scipy.optimize.minimize` to optimize the
        signal.

        Note. The error signal is normalized by the target to obtain
        reasonable error value during the optimization.
        """
        self.microscope.fluo.sigma_abs = {
            int(self.microscope.excitation.lambda_ * 1e9) : sigma_abs * 1e-20,
            int(self.microscope.sted.lambda_ * 1e9): self.microscope.fluo.sigma_abs[int(self.microscope.sted.lambda_ * 1e9)],
        }

        signal = self.expected_confocal_signal(p_ex, 0., pdt)
        error = ((target - signal) / target) ** 2

        return error

class Criterion:
    """
    Implements a `Criterion` that can be used to optimize the parameters of
    fluorescence
    """
    def __init__(self, criterions):
        """
        Instantiates the `Criterion`

        :param criterions: A `dict` of criterions
        """
        self.criterions = criterions

    def get(self, item, default=None):
        if item not in self.criterions:
            return default
        return self.criterions[item]

    def __contains__(self, item):
        return item in self.criterions

    def __str__(self):
        return str(self.criterions)

class ChoiceCriterion(Criterion):
    """
    Implements a `RandomCriterion` that can be used to optimize the parameters of
    fluorescence. This criterion allows to randomly sample from a range of
    parameters.
    """
    def __init__(self, criterions):
        super().__init__(criterions)
        self.criterions = {
            key : self.sample(value) for key, value in self.criterions.items()
        }

    def sample(self, criterion):
        """
        Samples from the given criterions

        :param criterion: A `dict` parameters and values to sample from
        """
        criterion = criterion.copy()
        for key, values in criterion.items():
            if isinstance(values, (tuple, list)):
                possible = numpy.linspace(*values, 5)
                criterion[key] = random.choice(possible).item()
            else:
                criterion[key] = values
        return criterion

class UniformCriterion(Criterion):
    """
    Implements a `RandomCriterion` that can be used to optimize the parameters of
    fluorescence. This criterion allows to randomly sample from a range of
    parameters.
    """
    def __init__(self, criterions):
        super().__init__(criterions)
        self.criterions = {
            key : self.sample(value) for key, value in self.criterions.items()
        }

    def sample(self, criterion):
        """
        Samples from the given criterions

        :param criterion: A `dict` parameters and values to sample from
        """
        criterion = criterion.copy()
        for key, values in criterion.items():
            if isinstance(values, (tuple, list)):
                criterion[key] = random.uniform(*values)
            else:
                criterion[key] = values
        return criterion
