
import numpy

from skimage import filters

from pysted import base, utils

def get_foreground(img):
    """Return a background mask of the given image using the OTSU method to threshold.

    :param 2d-array img: The image.

    :returns: A mask (2d array of bool: True on foreground, False on background).
    """
    val = filters.threshold_otsu(img)
    return img > val

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
        self.laser_ex_params = kwargs.get("laser_ex", {"lambda_" : 488e-9})
        self.laser_sted_params = kwargs.get("laser_sted", {"lambda_" : 575e-9, "zero_residual" : 0})
        self.detector_params = kwargs.get("detector", {"noise" : True})
        self.objective_params = kwargs.get("objective", {})
        self.fluo_params = kwargs.get("fluo",{
            "lambda_": 535e-9,
            "qy": 0.6,
            "sigma_abs": {488: 3e-20,
                          575: 6e-21},
            "sigma_ste": {560: 1.2e-20,
                          575: 6.0e-21,
                          580: 5.0e-21},
            "sigma_tri": 1e-21,
            "tau": 3e-09,
            "tau_vib": 1.0e-12,
            "tau_tri": 5e-6,
            "phy_react": {488: 0.25e-7,   # 1e-4
                          575: 25.0e-11},   # 1e-8
            "k_isc": 0.26e+6
        })
        self.pixelsize = 20e-9

    def generate_microscope(self, **kwargs):

        # Generating objects necessary for acquisition simulation
        laser_ex = base.GaussianBeam(**self.laser_ex_params)
        laser_sted = base.DonutBeam(**self.laser_sted_params)
        detector = base.Detector(**self.detector_params)
        objective = base.Objective(**self.objective_params)
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

    def generate_params(self, **kwargs):

        imaging_params = kwargs.get("imaging", {
            "pdt" : 100e-6,
            "p_ex" : 2e-6,
            "p_sted" : 0.
        })
        return imaging_params
