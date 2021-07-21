
import numpy
import os
import json

import gym_sted

from .prefNet import PrefNet

class PreferenceArticulator:
    """
    Creates a preference articulation model
    """
    def __init__(self, nb_obj=3):
        model_path = os.path.join(os.path.dirname(gym_sted.__file__), "prefnet", "2021-07-12-06-40-29_ResolutionBleachSNR")

        self.model = PrefNet(nb_obj=nb_obj)
        self.model = self.model.loading(os.path.join(model_path, "weights.t7"))
        self.model.eval()
        self.config = json.load(open(os.path.join(model_path, "config.json"), "r"))

    def articulate(self, thetas):
        """
        Articulates the decision from a list of possible choices

        :param thetas: A `list` of rewards

        :returns : An `int` of the optimal choice
        """
        # Converts to numpy ndarray and resize
        thetas = numpy.array(thetas)

        # Rescales the data appropriately
        thetas = (thetas - self.config["train_mean"]) / self.config["train_std"]

        # Predicts the objectives
        scores = self.model.predict(thetas)

        return numpy.argmax(scores)
