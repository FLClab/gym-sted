
import numpy 
import abberior
import yaml
import os

from abberior import microscope
from .general import GeneralMicroscope

class AbberiorConfigurator:
    def __init__(self, config, measurements):

        self.config = config
        self.measurements = measurements

    def initialize(self):
        """
        Sets the defaults value of the microscope
        """
        for key, conf in self.measurements.items():
            microscope.set_imagesize(
                conf, *self.config["image_opts"]["imagesize"])
            microscope.set_pixelsize(
                conf, *self.config["image_opts"]["pixelsize"])
            
            microscope.set_dwelltime(
                conf, self.config[f"params_{key}"]["pdt"])
            microscope.set_power(
                conf, self.config[f"params_{key}"]["p_sted"], laser_id=self.config["laser_id"]["sted"], channel_id=0)
            microscope.set_power(
                conf, self.config[f"params_{key}"]["p_ex"], laser_id=self.config["laser_id"]["exc"], channel_id=0)
        
    def set_params(self, measurement, params):
        """
        Sets the parameters of a measurement.

        :param measurement: A `str` of the required measurement 
        :param params: A `dict` of parameters
        """
        conf = self.measurements[measurement]

        microscope.set_dwelltime(
            conf, params.get("pdt", self.config[f"params_{measurement}"]["pdt"]))
        microscope.set_power(
            conf, params.get("p_sted", self.config[f"params_{measurement}"]["p_sted"]), laser_id=self.config["laser_id"]["sted"], channel_id=0)
        microscope.set_power(
            conf, params.get("p_ex", self.config[f"params_{measurement}"]["p_ex"]), laser_id=self.config["laser_id"]["exc"], channel_id=0)
            
class AbberiorMicroscope(GeneralMicroscope):
    def __init__(self, measurements, config=None):
        """
        :param measurements: A `dict` of measurement windows
        :param config: A `dict` of configuration options
        """

        if isinstance(config, type(None)):
            path = os.path.join(os.path.dirname(__file__), "configs", "default-abberior-config.yml")
            config = yaml.load(open(path, "r"), Loader=yaml.FullLoader)

        super().__init__(config)

        self.measurements = measurements
        self.configurator = AbberiorConfigurator(self.config, self.measurements)
        self.configurator.initialize()

    def acquire(self, measurement, params=None):

        if isinstance(params, dict):
            self.configurator.set_params(measurement, params)

        image, _ = abberior.microscope.acquire(
            self.measurements[measurement]
        )
        return numpy.array(image)[0][0]