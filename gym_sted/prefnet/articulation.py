
import numpy
import os
import json
import pickle
import bz2
import torch

import gym_sted

from .prefNet import PrefNet

def load_demonstrations(path=None, f1_score=False):
    """
    Loads a set of demonstrations from the given path

    :param path: A `str` of the path to the set of demonstrations

    :returns : A `list` of demonstrations
    """
    if isinstance(path, type(None)):
        path = os.path.join(os.path.dirname(gym_sted.__file__), "prefnet", "demonstrations", "demonstrations.pbz2")
    with bz2.open(path, "rb") as file:
        demonstrations = pickle.load(file)
    if f1_score:
        f1_scores = []
        for demonstration in demonstrations:
            f1_scores.extend([info["f1-score"] for info in demonstration])
        return f1_scores
    else:
        mo_objs = []
        for demonstration in demonstrations:
            mo_objs.extend([info["mo_objs"] for info in demonstration])
        return mo_objs

class PreferenceArticulator:
    """
    Creates a preference articulation model
    """
    def __init__(self, nb_obj=3, **kwargs):
        model_name = kwargs.get("model_name", "2021-07-12-06-40-29_ResolutionBleachSNR")
        model_path = os.path.join(os.path.dirname(gym_sted.__file__), "prefnet", model_name)

        self.model = PrefNet(nb_obj=nb_obj)
        self.model = self.model.loading(os.path.join(model_path, "weights.t7"))
        self.model.eval()
        self.config = json.load(open(os.path.join(model_path, "config.json"), "r"))

    def articulate(self, thetas, use_sigmoid=False):
        """
        Articulates the decision from a list of possible choices

        :param thetas: A `list` of rewards

        :returns : An `int` of the optimal choice
                   An `numpy.ndarray` of the scores from low to high
        """
        # Converts to numpy ndarray and resize
        thetas = numpy.array(thetas)

        # Rescales the data appropriately
        thetas = (thetas - self.config["train_mean"]) / self.config["train_std"]

        # Predicts the objectives
        scores = self.model.predict(thetas)
        if use_sigmoid:
            scores = torch.sigmoid(torch.tensor(scores))
            scores = scores.cpu().data.numpy()

        # Sorts the scores
        sorted_scores = numpy.argsort(scores.ravel())
        return scores, sorted_scores[-1], sorted_scores
