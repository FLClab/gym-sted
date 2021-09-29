
import numpy
import pickle
import os

from collections import defaultdict
from matplotlib import pyplot

from gym_sted.prefnet import PreferenceArticulator

PATH = "/home-local/gym-sted"

class LeaderBoard:
    """
    Creates a `LeaderBoard`
    """
    def __init__(self, models, bleach="low-bleach"):

        self.models = models

        self.bleach = bleach
        self.num_episodes = 100
        self.num_steps = 10
        self.preference_articulator = PreferenceArticulator()

    def get_model_records(self, model):
        """
        Loads the model records from the model folder

        :param model: A `str` of the model name

        :returns : A `list` of records
        """
        records = pickle.load(open(os.path.join(PATH, model, "eval", "stats.pkl"), "rb"))
        return records[self.bleach]

    def get_leaderboard(self):
        """
        Gets the leaderboard from the models

        :returns : A `dict` of scores
        """
        boards = {}
        for model in self.models:
            boards[model] = self.get_model_records(model)

        episode_articulations = []
        for episode in range(self.num_episodes):
            step_articulations = []
            for step in range(self.num_steps):
                mo_objs = []
                for model in self.models:
                    mo_objs.append(boards[model][episode][step]["mo_objs"])
                _, articulation = self.preference_articulator.articulate(mo_objs)
                # Articulation is sorted with increasing scores, we need to inverse
                articulation = numpy.abs((len(articulation) - 1) - articulation)
                step_articulations.append(articulation)
            episode_articulations.append(step_articulations)
        episode_articulations = numpy.array(episode_articulations)

        board = {
            "top1" : self.get_topx(episode_articulations, 1),
            "top3" : self.get_topx(episode_articulations, 3)
        }
        return board

    def get_topx(self, articulations, x):
        """
        Computes the number of times a particular model is top x

        :param articulations: A `numpy.ndarray` of the articulations with shape [num_episodes, num_steps, num_models]
        :param x: An `int` of the top

        :returns : A `numpy.ndarray` of the ratio of top x
        """
        top = articulations < x
        top = numpy.sum(top, axis=0) / len(top)
        return top

def plot_top(board, key, models=None, *args, **kwargs):
    """
    Creates a graph of the top1 across the steps

    :param board: A `dict` of the board
    :param models: (optional) A `list` of model names

    :returns : A `matplotlib.Figure`
               A `matplotlib.Axes`
    """
    if isinstance(models, type(None)):
        models = [f"{i}" for i in range(board[key].shape[1])]
    fig, ax = pyplot.subplots()
    for i, model in enumerate(models):
        x = numpy.arange(board[key].shape[0])
        ax.plot(x, board[key][:, i], label=model)
    ax.set(
        **kwargs
    )
    pyplot.legend()
    return fig, ax

if __name__ == "__main__":

    models = [
        "20210903-114835_a3972c26",
        "20210903-115820_e364ff60",
        "20210903-115906_67980011",
        "20210914-122242_25874d69",
        "20210914-122242_7b6e681c",
        "20210914-122242_f5426684",
    ]

    leaderboard = LeaderBoard(models, bleach="low-bleach")
    board = leaderboard.get_leaderboard()
    print(board)

    fig, ax = plot_top(
        board, "top1", models=models,
        xlabel="Steps", ylabel="Ranking", ylim=(0, 1), title="Top-1"
    )
    fig, ax = plot_top(
        board, "top3", models=models,
        xlabel="Steps", ylabel="Ranking", ylim=(0, 1), title="Top-3"
    )
    pyplot.show()
