
import numpy 
import abberior
import yaml
import os

from abberior import microscope
from .general import GeneralMicroscope

class AbberiorConfigurator:
    """
    Configuration of an Abberior microscope.

    This class is used to configure the microscope settings for a given measurement.
    """
    def __init__(self, config, measurements):
        """
        Instantiates the `AbberiorConfigurator`

        :param config: A `dict` of configuration options
        :param measurements: A `dict` of measurement windows
        """

        self.config = config
        self.measurements = measurements

    def get_laser_id(self, measurement_name):
        """
        Returns the laser id for a given measurement.

        :param measurement_name: A `str` of the measurement name

        :return: A `dict` of laser ids
        """
        if self.config.get("laser_id", None) is None:
            exc_laser_id = self.config[f"params_{measurement_name}"]["exc_laser_id"]
            sted_laser_id = self.config[f"params_{measurement_name}"]["sted_laser_id"]
        else:
            exc_laser_id = self.config["laser_id"]["exc"]
            sted_laser_id = self.config["laser_id"]["sted"]

        return {"exc": exc_laser_id, "sted": sted_laser_id}

    def initialize(self):
        """
        Sets the defaults value of the microscope
        """
        for key, conf in self.measurements.items():
            microscope.set_imagesize(
                conf, *self.config["image_opts"]["imagesize"])
            microscope.set_pixelsize(
                conf, *self.config["image_opts"]["pixelsize"])
            
            laser_id = self.get_laser_id(key)
            microscope.set_dwelltime(
                conf, self.config[f"params_{key}"]["pdt"])
            microscope.set_power(
                conf, self.config[f"params_{key}"]["p_sted"], laser_id=laser_id["sted"], channel_id=0)
            microscope.set_power(
                conf, self.config[f"params_{key}"]["p_ex"], laser_id=laser_id["exc"], channel_id=0)
        
    def set_params(self, measurement, params):
        """
        Sets the parameters of a measurement.

        :param measurement: A `str` of the required measurement 
        :param params: A `dict` of parameters
        """
        conf = self.measurements[measurement]

        laser_id = self.get_laser_id(measurement)

        microscope.set_dwelltime(
            conf, params.get("pdt", self.config[f"params_{measurement}"]["pdt"]))
        microscope.set_power(
            conf, params.get("p_sted", self.config[f"params_{measurement}"]["p_sted"]), laser_id=laser_id["sted"], channel_id=0)
        microscope.set_power(
            conf, params.get("p_ex", self.config[f"params_{measurement}"]["p_ex"]), laser_id=laser_id["exc"], channel_id=0)
            
class AbberiorMicroscope(GeneralMicroscope):
    """
    Superclass for an Abberior microscope.

    This class implements the `acquire` method to acquire images from an Abberior microscope.
    """
    def __init__(self, measurements, config=None):
        """
        Instantiates the `AbberiorMicroscope`

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
        """
        Acquires an image from the microscope.

        :param measurement: A `str` of the required measurement
        :param params: A `dict` of parameters

        :return: A `numpy.ndarray` of the acquired image
        """
        if isinstance(params, dict):
            self.configurator.set_params(measurement, params)

        image, _ = abberior.microscope.acquire(
            self.measurements[measurement]
        )
        return numpy.array(image)[0][0]