
import numpy

class RewardCalculator:
    def __init__(self, objs, *args, **kwargs):
        """
        Instantiates the `RewardCalculator`

        :param objs: A `dict` of objective
        """
        self.objectives = objs

    def evaluate(self, sted_image, conf1, conf2, fg_s, fg_c):
        """
        Evaluates the objectives and returns whether all objectives are within
        the predefined bounds

        :param sted_stack: A `numpy.ndarray` of the acquired STED image
        :param conf1: A `numpy.ndarray` of the acquired confocal image before
        :param conf2: A `numpy.ndarray` of the acquired confocal image after
        :param fg_s: A `numpy.ndarray` of the STED foreground
        :param fg_c: A `numpy.ndarray` of the confocal foreground

        :returns : A `list` of rewards
        """
        raise NotImplementedError

    def rescale(self, value, obj_name):
        """
        Rescales the reward to be within approximately [0, 1] range

        :param value: The value of the objective
        :param obj_name: The name of the objective
        """
        return (value - self.scales[obj_name]["low"]) / (self.scales[obj_name]["high"] - self.scales[obj_name]["low"])

class MORewardCalculator(RewardCalculator):
    def __init__(self, objs, *args, **kwargs):
        """
        Instantiates the `RewardCalculator`

        :param objs: A `dict` of objective
        """
        super().__init__(objs)
        self.scales = kwargs.get("scales", None)

    def evaluate(self, sted_image, conf1, conf2, fg_s, fg_c):
        """
        Evaluates the objectives and returns all objectives

        :param sted_stack: A `numpy.ndarray` of the acquired STED image
        :param conf1: A `numpy.ndarray` of the acquired confocal image before
        :param conf2: A `numpy.ndarray` of the acquired confocal image after
        :param fg_s: A `numpy.ndarray` of the STED foreground
        :param fg_c: A `numpy.ndarray` of the confocal foreground

        :returns : A `list` of rewards
        """
        return [
            self.objectives[obj_name].evaluate([sted_image], conf1, conf2, fg_s, fg_c)
            for obj_name in self.objectives.keys()
        ]

    def evaluate_rescale(self, sted_image, conf1, conf2, fg_s, fg_c):
        """
        Evaluates the objectives and returns whether all objectives are within
        the predefined bounds

        :param sted_stack: A `numpy.ndarray` of the acquired STED image
        :param conf1: A `numpy.ndarray` of the acquired confocal image before
        :param conf2: A `numpy.ndarray` of the acquired confocal image after
        :param fg_s: A `numpy.ndarray` of the STED foreground
        :param fg_c: A `numpy.ndarray` of the confocal foreground

        :returns : A `list` of rewards
        """
        return [
            self.rescale(self.objectives[obj_name].evaluate([sted_image], conf1, conf2, fg_s, fg_c), obj_name)
            for obj_name in self.objectives.keys()
        ]

class NanodomainsRewardCalculator(RewardCalculator):
    def __init__(self, objs, *args, **kwargs):
        """
        Instantiates the `NanodomainsRewardCalculator`

        :param objs: A `dict` of objective
        """
        super().__init__(objs)
        self.scales = kwargs.get("scales")

    def evaluate(self, sted_stack, confocal_init, confocal_end, sted_fg, confocal_fg, threshold=2, *args, **kwargs):
        """
        Evaluates the objectives and returns all objectives

        :param sted_stack: A list of STED images.
        :param confocal_init: A confocal image acquired before the STED stack.
        :param sted_fg: A background mask of the first STED image in the stack
                        (2d array of bool: True on foreground, False on background).
        :param confocal_fg: A background mask of the initial confocal image
                            (2d array of bool: True on foreground, False on background).

        :returns : A `float` of the reward
        """
        return self.rescale(self.objectives["NbNanodomains"].evaluate(
                sted_stack, confocal_init, confocal_end, sted_fg, confocal_fg, *args, **kwargs),
                "NbNanodomains")
