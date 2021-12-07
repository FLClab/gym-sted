
import numpy
import os
import json
import pickle
import bz2

import gym_sted

def load_demonstrations(path=None):
    """
    Loads a set of demonstrations from the given path

    :param path: A `str` of the path to the set of demonstrations

    :returns : A `list` of demonstrations
    """
    if isinstance(path, type(None)):
        path = os.path.join(os.path.dirname(gym_sted.__file__), "prefnet", "demonstrations", "demonstrations.pbz2")
    with bz2.open(path, "rb") as file:
        demonstrations = pickle.load(file)
    f1_scores = []
    for demonstration in demonstrations:
        f1_scores.extend([info["f1-score"] for info in demonstration])
    return f1_scores
