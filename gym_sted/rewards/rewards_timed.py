"""
***
This implementation is for the rewards of the 2nd gym setting, in which datamaps evolve over time. Some details, such as
bleaching computation, differ due to this, which is why the implementation here differs from the one in objectives.py
***
"""

import numpy


class RewardCalculator:
    def __init__(self, objs, *args, **kwargs):
        """
        Instantiates the `RewardCalculator`

        :param objs: A `dict` of objective
        """
        self.objectives = objs

    def evaluate(self, sted_stack, confocal_init, sted_fg, confocal_fg, n_molecs_init, n_molecs_post, temporal_datamap,
                 threshold=2):
        """
        Evaluates the objectives and returns whether all objectives are within
        the predefined bounds

        :param sted_stack: A list of STED images.
        :param confocal_init: A confocal image acquired before the STED stack.
        :param sted_fg: A background mask of the first STED image in the stack
                        (2d array of bool: True on foreground, False on background).
        :param confocal_fg: A background mask of the initial confocal image
                            (2d array of bool: True on foreground, False on background).
        :param n_molecs_init: The number of molecules in the BASE DATAMAP before the acquisition
        :param n_molecs_post: The number of molecules in the BASE DATAMAP after the acquisition
        :param temporal_datamap: The temporal_datamap object being acquired on, containing a synapse object with info on
                                 the number and positions of the nanodomains

        :returns : A `list` of rewards
        """
        raise NotImplementedError


class MORewardCalculator(RewardCalculator):
    def __init__(self, objs, *args, **kwargs):
        """
        ***
        I think this is the only rewards class I want. This will return a vector with the reward values for all 4
        objectives. I will only use the reward value for Nanodomains identification as the explicit reward signal,
        and use the 3 other objectives as info in the encoded representation of the Neural Net (or something)
        ??? How do I know which idx in the returned list corresponds to which objective ???
        ***

        Instantiates the `RewardCalculator`

        :param objs: A `dict` of objective
        """
        super().__init__(objs)
        self.scales = kwargs.get("scales")

    def evaluate(self, sted_stack, confocal_init, sted_fg, confocal_fg, n_molecs_init, n_molecs_post, temporal_datamap,
                 threshold=2):
        """
        Evaluates the objectives and returns whether all objectives are within
        the predefined bounds

        :param sted_stack: A list of STED images.
        :param confocal_init: A confocal image acquired before the STED stack.
        :param sted_fg: A background mask of the first STED image in the stack
                        (2d array of bool: True on foreground, False on background).
        :param confocal_fg: A background mask of the initial confocal image
                            (2d array of bool: True on foreground, False on background).
        :param n_molecs_init: The number of molecules in the BASE DATAMAP before the acquisition
        :param n_molecs_post: The number of molecules in the BASE DATAMAP after the acquisition
        :param temporal_datamap: The temporal_datamap object being acquired on, containing a synapse object with info on
                                 the number and positions of the nanodomains

        :returns : A `list` of rewards
        """
        # maybe it would be better to make it a dict instead?
        return {
            obj_name: 1 - self.rescale(self.objectives[obj_name].evaluate(sted_stack, confocal_init, sted_fg,
                                                                          confocal_fg, n_molecs_init, n_molecs_post,
                                                                          temporal_datamap),
                                       obj_name)
            if obj_name != "SNR" and obj_name != "NbNanodomains" else self.rescale(self.objectives[obj_name].evaluate(
                sted_stack, confocal_init, sted_fg, confocal_fg, n_molecs_init, n_molecs_post, temporal_datamap),
                obj_name)
            for obj_name in self.objectives.keys()
        }
        # return [
        #     1 - self.rescale(self.objectives[obj_name].evaluate(sted_stack, confocal_init, sted_fg, confocal_fg,
        #                                                         n_molecs_init, n_molecs_post, temporal_datamap),
        #                      obj_name)
        #     if obj_name != "SNR" and obj_name != "NbNanodomains" else self.rescale(self.objectives[obj_name].evaluate(
        #         sted_stack, confocal_init, sted_fg, confocal_fg, n_molecs_init, n_molecs_post, temporal_datamap),
        #         obj_name)
        #     for obj_name in self.objectives.keys()
        # ]

    def rescale(self, value, obj_name):
        """
        Rescales the reward to be within approximately [0, 1] range

        :param value: The value of the objective
        :param obj_name: The name of the objective
        """
        return (value - self.scales[obj_name]["min"]) / (self.scales[obj_name]["max"] - self.scales[obj_name]["min"])


class NanodomainsRewardCalculator(RewardCalculator):
    def __init__(self, objs, *args, **kwargs):
        """
        ***
        I think this is the only rewards class I want. This will return a vector with the reward values for all 4
        objectives. I will only use the reward value for Nanodomains identification as the explicit reward signal,
        and use the 3 other objectives as info in the encoded representation of the Neural Net (or something)
        ??? How do I know which idx in the returned list corresponds to which objective ???
        ***

        Instantiates the `RewardCalculator`

        :param objs: A `dict` of objective
        """
        super().__init__(objs)
        self.scales = kwargs.get("scales")

    def evaluate(self, sted_stack, confocal_init, sted_fg, confocal_fg, n_molecs_init, n_molecs_post, temporal_datamap,
                 threshold=2):
        """
        Evaluates the objectives and returns whether all objectives are within
        the predefined bounds

        :param sted_stack: A list of STED images.
        :param confocal_init: A confocal image acquired before the STED stack.
        :param sted_fg: A background mask of the first STED image in the stack
                        (2d array of bool: True on foreground, False on background).
        :param confocal_fg: A background mask of the initial confocal image
                            (2d array of bool: True on foreground, False on background).
        :param n_molecs_init: The number of molecules in the BASE DATAMAP before the acquisition
        :param n_molecs_post: The number of molecules in the BASE DATAMAP after the acquisition
        :param temporal_datamap: The temporal_datamap object being acquired on, containing a synapse object with info on
                                 the number and positions of the nanodomains

        :returns : A `list` of rewards
        """
        # maybe it would be better to make it a dict instead?
        return self.rescale(self.objectives["NbNanodomains"].evaluate(
                sted_stack, confocal_init, sted_fg, confocal_fg, n_molecs_init, n_molecs_post, temporal_datamap),
                "NbNanodomains")

    def rescale(self, value, obj_name):
        """
        Rescales the reward to be within approximately [0, 1] range

        :param value: The value of the objective
        :param obj_name: The name of the objective
        """
        return (value - self.scales[obj_name]["min"]) / (self.scales[obj_name]["max"] - self.scales[obj_name]["min"])
