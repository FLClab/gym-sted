
import numpy

class RewardCalculator:
    """
    Base class for the reward calculator
    """
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
        Rescales the reward to be within ["low", "high"] range

        :param value: The value of the objective
        :param obj_name: The name of the objective
        """
        return (value - self.scales[obj_name]["low"]) / (self.scales[obj_name]["high"] - self.scales[obj_name]["low"])

class MORewardCalculator(RewardCalculator):
    """
    Multi-objective reward calculator

    The `MORewardCalculator` is used to evaluate multiple objectives and returns the rewards as a list of values for each imaging objective.
    """
    def __init__(self, objs, *args, **kwargs):
        """
        Instantiates the `RewardCalculator`

        :param objs: A `dict` of objective
        """
        super().__init__(objs)
        self.scales = kwargs.get("scales", None)

    def evaluate(self, sted_image, conf1, conf2, fg_s, fg_c):
        """
        Evaluates the objectives

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
        Evaluates the objectives with rescaling

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

class SumRewardCalculator(RewardCalculator):
    """
    Sum reward calculator

    The `SumRewardCalculator` is used to evaluate multiple objectives and returns the sum of the rewards.
    """
    def __init__(self, objs, *args, **kwargs):
        """
        Instantiates the `RewardCalculator`

        :param objs: A `dict` of objective
        """
        super().__init__(objs)
        self.scales = kwargs.get("scales")

    def evaluate(self, sted_image, conf1, conf2, fg_s, fg_c):
        """
        Evaluates the objectives and returns the sum of the rewards

        :param sted_stack: A `numpy.ndarray` of the acquired STED image
        :param conf1: A `numpy.ndarray` of the acquired confocal image before
        :param conf2: A `numpy.ndarray` of the acquired confocal image after
        :param fg_s: A `numpy.ndarray` of the STED foreground
        :param fg_c: A `numpy.ndarray` of the confocal foreground

        :returns : A `float` of the sum of the rewards
        """
        return sum([
            1 - self.rescale(self.objectives[obj_name].evaluate([sted_image], conf1, conf2, fg_s, fg_c), obj_name)
            if obj_name != "SNR" else self.rescale(self.objectives[obj_name].evaluate([sted_image], conf1, conf2, fg_s, fg_c), obj_name)
            for obj_name in self.objectives.keys()
        ])

class MultiplyRewardCalculator(RewardCalculator):
    """
    Multiply reward calculator

    The `MultiplyRewardCalculator` is used to evaluate multiple objectives and returns the product of the rewards.
    """
    def __init__(self, objs, *args, **kwargs):
        """
        Instantiates the `RewardCalculator`

        :param objs: A `dict` of objective
        """
        super().__init__(objs)
        self.scales = kwargs.get("scales")

    def evaluate(self, sted_image, conf1, conf2, fg_s, fg_c):
        """
        Evaluates the objectives and returns the product of the rewards

        :param sted_stack: A `numpy.ndarray` of the acquired STED image
        :param conf1: A `numpy.ndarray` of the acquired confocal image before
        :param conf2: A `numpy.ndarray` of the acquired confocal image after
        :param fg_s: A `numpy.ndarray` of the STED foreground
        :param fg_c: A `numpy.ndarray` of the confocal foreground

        :returns : A `float` of the product of the rewards
        """
        return numpy.prod([
            max(0, 1 - self.rescale(self.objectives[obj_name].evaluate([sted_image], conf1, conf2, fg_s, fg_c), obj_name))
            if obj_name != "SNR" else self.rescale(self.objectives[obj_name].evaluate([sted_image], conf1, conf2, fg_s, fg_c), obj_name)
            for obj_name in self.objectives.keys()
        ]).item()

class BoundedRewardCalculator(RewardCalculator):
    """
    Bound reward calculator

    The `BoundedRewardCalculator` is used to evaluate multiple objectives and returns whether all objectives are within the predefined bounds.
    """
    def __init__(self, objs, *args, **kwargs):
        """
        Instantiates the `RewardCalculator`

        :param objs: A `dict` of objective
        :param bounds: A `dict` of the bounds for each objective
        """
        super().__init__(objs)
        self.bounds = kwargs.get("bounds")

    def evaluate(self, sted_image, conf1, conf2, fg_s, fg_c):
        """
        Evaluates the objectives and returns whether all objectives are within
        the predefined bounds

        :param sted_stack: A `numpy.ndarray` of the acquired STED image
        :param conf1: A `numpy.ndarray` of the acquired confocal image before
        :param conf2: A `numpy.ndarray` of the acquired confocal image after
        :param fg_s: A `numpy.ndarray` of the STED foreground
        :param fg_c: A `numpy.ndarray` of the confocal foreground

        :returns : An `int` in  `{0, 1}` if all objectives are within bounds
        """
        return int(all([
            self.isin(self.objectives[obj_name].evaluate([sted_image], conf1, conf2, fg_s, fg_c), obj_name)
            for obj_name in self.objectives.keys()
        ]))

    def isin(self, value, obj_name):
        """
        Evaluates wheter the specific value is within the bounds

        :param value: The value of the objective
        :param obj_name: The name of the objetive for bound checking

        :returns : A `bool` whether the value is within the bounds
        """
        return self.bounds[obj_name]["low"] < value < self.bounds[obj_name]["high"]

class NanodomainsRewardCalculator(RewardCalculator):
    """
    Nano-domains reward calculator

    The `NanodomainsRewardCalculator` is used to evaluate the number of nanodomains that were correctly identified in the STED image.
    """
    def __init__(self, objs, *args, **kwargs):
        """
        Instantiates the `RewardCalculator`

        :param objs: A `dict` of objective
        """
        super().__init__(objs)
        self.scales = kwargs.get("scales")

    def evaluate(self, sted_stack, confocal_init, confocal_end, sted_fg, confocal_fg, threshold=2, *args, **kwargs):
        """
        Evaluates the objectives and returns whether all objectives are within
        the predefined bounds

        :param sted_stack: A list of STED images.
        :param confocal_init: A confocal image acquired before the STED stack.
        :param sted_fg: A background mask of the first STED image in the stack
                        (2d array of bool: True on foreground, False on background).
        :param confocal_fg: A background mask of the initial confocal image
                            (2d array of bool: True on foreground, False on background).

        :returns : A `list` of rewards
        """
        # maybe it would be better to make it a dict instead?
        return self.rescale(self.objectives["NbNanodomains"].evaluate(
                sted_stack, confocal_init, confocal_end, sted_fg, confocal_fg, *args, **kwargs),
                "NbNanodomains")

