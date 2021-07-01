
import numpy

class RewardCalculator:
    def __init__(self, objs):
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
        return [
            self.objectives[obj_name].evaluate([sted_image], conf1, conf2, fg_s, fg_c), obj_name)
            for obj_name in self.objectives.keys()
        ]

class BoundedRewardCalculator:
    def __init__(self, objs, bounds):
        """
        Instantiates the `RewardCalculator`

        :param objs: A `dict` of objective
        :param bounds: A `dict` of the bounds for each objective
        """
        self.objectives = objs
        self.bounds = bounds

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
        return self.bounds[obj_name]["min"] <= value <= self.bounds[obj_name]["max"]
