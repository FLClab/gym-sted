import numpy 
import os, glob
import json
import random

from skimage import filters
from collections import defaultdict

class DatamapGenerator:
    """
    Creates a `DatamapGenerator` that allows to create a datamap

    Only one mode is available:
    - `real`: Samples from real datamaps
    """
    def __init__(
        self, mode="real", shape=(224, 224),
        molecules=(10, 100), molecules_scale=0.1, random_state=None, 
        path=None, augment=False
    ):
        """
        Instantiates the `DatamapGenerator`

        :param mode: A `str` that specifies the mode of the datamap generator
        :param shape: A `tuple` of the shape of the datamap
        :param molecules: An `int` or a `tuple` of the number of molecules to sample
        :param molecules_scale: A `float` of the standard deviation of the number of molecules
        :param random_state: An `int` that sets the random state
        :param path: A `str` to the folder where the datamaps are stored
        :param augment: A `bool` that specifies if the datamap should be augmented
        """
        self.mode = mode
        assert self.mode in ["real"]
        self.molecules = molecules
        self.molecules_scale = molecules_scale
        self.shape = shape
        self.idx = None
        self.augment = augment

        self.path = path
        if isinstance(self.path, str):
            self.path = [self.path]
        if isinstance(self.path, (list, tuple)) and ("real" in self.mode):
            self.datamaps = []
            for path in self.path:
                exclude = os.path.join(path, "exclude.json")
                files = sorted(glob.glob(os.path.join(path, "*.npy")))
                if os.path.isfile(exclude):
                    exclude_files = json.load(open(exclude, "r"))
                    files = list(filter(lambda file: not any([exclude in file for exclude in exclude_files]), files))
                self.datamaps.extend(files)
            self.mode = "real"

        self.random = numpy.random.RandomState(random_state)
        self.groups = [os.path.basename(path) for path in self.path]
        self.grouped_datamaps = defaultdict(list)
        for datamap in self.datamaps:
            group = os.path.basename(os.path.dirname(datamap))
            self.grouped_datamaps[group].append(datamap)

    def sample_group(self):
        """
        Samples from the available groups
        """
        group = numpy.random.choice(self.groups)
        return group

    def __call__(self, **kwargs):
        """
        Implements a generic `__call__` method
        """
        if isinstance(self.molecules, (list, tuple)):
            molecules = self.random.randint(*self.molecules)
            # molecules = self.molecules[0] if self.random.rand() > 0.5 else self.molecules[1]
        else:
            molecules = self.molecules
            scale = self.random.normal(loc=molecules, scale=self.molecules_scale * molecules)
            while (scale < 1) or (scale > molecules):
                scale = self.random.normal(loc=molecules, scale=self.molecules_scale * molecules)
            molecules = int(scale)
        return getattr(self, f"_{self.mode}_datamap_generator")(
            molecules = molecules,
            **kwargs
        )

    def _real_datamap_generator(self, molecules, group=None, *args, **kwargs):
        """
        Methods that allows to use the datamaps that were generated from real
        experiments by a U-Net.

        :param molecules: An `int` of the average number of molecules to samples
        :param group: A `str` of the group to sample from

        :return: A `numpy.ndarray` of the datamap
        """
        idx = kwargs.get("idx", None)
        if isinstance(idx, int):
            datamap = self.datamaps[idx]
        else:
            if isinstance(group, str):
                idx = self.random.randint(len(self.grouped_datamaps[group]))
                self.idx = idx 
                datamap = self.grouped_datamaps[group][self.idx]
            else:
                idx = self.random.randint(len(self.datamaps))
                self.idx = idx
                datamap = self.datamaps[idx]

        datamap = numpy.load(datamap)

        if isinstance(self.shape, int):
            shape = (self.shape, self.shape)
        else:
            shape = self.shape

        # Pads the array in cases where the desired crop is bigger than the datamaps
        pady, padx = max(0, shape[0] - datamap.shape[0]), max(0, shape[1] - datamap.shape[1])
        datamap = numpy.pad(datamap, ((pady, pady), (padx, padx)), mode="symmetric")

        # Random crop of the datamap
        # We filter the datamap with a gaussian kernel to sample where the
        # molecules are
        sy, sx = shape[0] // 2, shape[1] // 2
        prob = filters.gaussian(datamap * (datamap >= 0.5).astype(int), sigma=10)[sy : -sy + 1, sx : -sx + 1] + 1e-9 # adds numerical stability
        prob = prob / prob.sum()
        choice = numpy.random.choice(prob.size, p=prob.ravel())

        # Multiplies by a random number of molecules
        # scale = self.random.normal(loc=molecules, scale=self.molecules_scale * molecules)
        # while (scale < 1) or (scale > molecules):
        #     scale = self.random.normal(loc=molecules, scale=self.molecules_scale * molecules)
        datamap *= int(molecules)

        j, i = numpy.unravel_index(choice, prob.shape)
        j += sy
        i += sx
        # j, i = self.random.randint(0, datamap.shape[0] - shape[0]), self.random.randint(0, datamap.shape[1] - shape[1])
        datamap = datamap[j - sy : j + sy, i - sx : i + sx]
        
        if self.augment:
            # 90 rotation
            if random.random() < 0.5:
                k = random.randint(1, 3)
                datamap = numpy.rot90(datamap, k, axes=(-2, -1))
            # Up-down flip
            if random.random() < 0.5: 
                datamap = numpy.flip(datamap, axis=-2)
            # Left-right flip
            if random.random() < 0.5: 
                datamap = numpy.flip(datamap, axis=-1)                

        return datamap