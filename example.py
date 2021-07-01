
import numpy
import argparse
import sys
import random
import gym

from gym import wrappers, logger, envs
from collections import OrderedDict
from tqdm.auto import trange

from gym_sted.rewards import RewardCalculator
from gym_sted.agents import RandomAgent
from gym_sted.rewards import objectives

obj_dict = {
    "SNR" : objectives.Signal_Ratio(75),
    "Bleach" : objectives.Bleach(),
    "Resolution" : objectives.Resolution(pixelsize=20e-9),
    "Squirrel" : objectives.Squirrel()
}
bounds_dict = {
    "SNR" : {"min" : 1, "max" : numpy.inf},
    "Bleach" : {"min" : -numpy.inf, "max" : 0.1},
    "Resolution" : {"min" : 0, "max" : 80},
    "Squirrel" : {"min" : 0, "max" : 12}
}

if __name__ == '__main__':

    obj_names = ["Resolution", "Bleach", "Squirrel"]

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='gym_sted:STED-v0', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    # logger.set_level(logger.INFO)
    env = gym.make(args.env_id)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    import tempfile
    outdir = './tmp/random-agent-results'
    outdir = tempfile.mkdtemp()
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = RandomAgent(env.action_space)

    # Update creates a reward calculator
    objs = OrderedDict({obj_name : obj_dict[obj_name] for obj_name in obj_names})
    bounds = OrderedDict({obj_name : bounds_dict[obj_name] for obj_name in obj_names})
    env.update_(
        reward_calculator=RewardCalculator(objs)
    )

    episode_count = 2
    reward = 0
    done = False
    render = False

    for i in range(episode_count):
        observation = env.reset()
        while True:
            action = agent.act(observation, reward, done)
            observation, reward, done, info = env.step(action)
            if render:
                # renders the acquired images
                env.render(info)
            if done:
                break

    # Close the env and write monitor result info to disk
    env.close()
