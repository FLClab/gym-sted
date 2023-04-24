
class GeneralMicroscope:
    """
    Implements a General Microscope. This class should be inherited
    by other microscopes
    """

    def __init__(self, config):

        self.config = config

    def get_conf_params():
        return {
            "p_ex" : self.config["conf-params"]["p_ex"],
            "pdt" : self.config["conf-params"]["pdt"],
        }

    def acquire_confocal(self):
        """
        Acquires a low-resolution image with the parameters defined 
        in the configuration file
        """
        params = self.get_conf_params()
        self.acquire(params)

    def acquire_sted(self, params):
        """
        Acquires a high-resolution image with the given parameters

        :param params: A `dict` of the parameters
        """
        self.acquire(params)

    def acquire(self, measurement, params):
        raise NotImplementedError