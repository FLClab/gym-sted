
import gym
import numpy
import random

from gym import error, spaces, utils
from gym.utils import seeding
from matplotlib import pyplot
from collections import OrderedDict

from gym_sted import rewards, defaults
from gym_sted.utils import SynapseGenerator, MicroscopeGenerator, get_foreground
from gym_sted.rewards import objectives

obj_dict = {
    "SNR" : objectives.Signal_Ratio(75),
    "Bleach" : objectives.Bleach(),
    "Resolution" : objectives.Resolution(pixelsize=20e-9),
    "Squirrel" : objectives.Squirrel()
}
bounds_dict = {
    "SNR" : {"min" : 0.20, "max" : numpy.inf},
    "Bleach" : {"min" : -numpy.inf, "max" : 0.5},
    "Resolution" : {"min" : 0, "max" : 100}
}
scales_dict = {
    "SNR" : {"min" : 0, "max" : 1},
    "Bleach" : {"min" : 0, "max" : 1},
    "Resolution" : {"min" : 40, "max" : 180}
}

class STEDEnv(gym.Env):
    """
    Creates a `STEDEnv`

    The `STEDEnv` implements a scan of the entire field of view
    """
    metadata = {'render.modes': ['human']}
    obj_names = ["Resolution", "Bleach", "SNR"]

    def __init__(self, reward_calculator="SumRewardCalculator", actions=["p_sted"]):

        self.synapse_generator = SynapseGenerator(mode="mushroom", seed=None)
        self.microscope_generator = MicroscopeGenerator()
        self.microscope = self.microscope_generator.generate_microscope()

        self.actions = actions
        self.action_space = spaces.Box(
            low=numpy.array([defaults.action_spaces[name]["low"] for name in self.actions]),
            high=numpy.array([defaults.action_spaces[name]["high"] for name in self.actions]),
            shape=(len(self.actions),), dtype=numpy.float32
        )
        self.observation_space = spaces.Tuple((
            spaces.Box(0, 2**16, shape=(64, 64, 1), dtype=numpy.uint16),
            spaces.Box(0, 1, shape=(len(self.obj_names) + len(self.actions),), dtype=numpy.float32)
        ))

        self.state = None
        self.initial_count = None
        self.current_step = 0

        objs = OrderedDict({obj_name : obj_dict[obj_name] for obj_name in self.obj_names})
        bounds = OrderedDict({obj_name : bounds_dict[obj_name] for obj_name in self.obj_names})
        scales = OrderedDict({obj_name : scales_dict[obj_name] for obj_name in self.obj_names})
        self.reward_calculator = getattr(rewards, reward_calculator)(objs, bounds=bounds, scales=scales)
        self.mo_reward_calculator = rewards.MORewardCalculator(objs, bounds=bounds, scales=scales)

        self.datamap = None
        self.viewer = None

        self.seed()

    def step(self, action):

        # We manually clip the actions which are out of action space
        action = numpy.clip(action, self.action_space.low, self.action_space.high)

        # Acquire the image
        sted_image, bleached, conf1, conf2, fg_s, fg_c = self._acquire(action)

        reward = self.reward_calculator.evaluate(sted_image, conf1, conf2, fg_s, fg_c)
        mo_objs = self.mo_reward_calculator.evaluate(sted_image, conf1, conf2, fg_s, fg_c)

        num_killed = (self.initial_count - self.datamap.whole_datamap.sum()) / self.initial_count
        # print(action / defaults.action_spaces["p_sted"]["high"], num_killed)
        done = (self.current_step >= self.spec.max_episode_steps) \
                or (num_killed > 0.90)
        # print(rewards, reward, done)
        observation = conf2[..., numpy.newaxis]
        info = {
            "action" : action,
            "bleached" : bleached,
            "sted_image" : sted_image,
            "conf1" : conf1,
            "conf2" : conf2,
            "fg_c" : fg_c,
            "fg_s" : fg_s,
            "mo_objs" : mo_objs
        }
        self.current_step += 1

        return (observation, numpy.array(mo_objs + action.tolist())), reward, done, info

    def reset(self):
        """
        Resets the environment with a new datamap

        :returns : A `numpy.ndarray` of the molecules
        """
        self.current_step = 0
        state = self._update_datamap()
        self.initial_count = self.datamap.whole_datamap.sum()

        self.state = state[..., numpy.newaxis]
        return (self.state, numpy.zeros((len(self.obj_names) + len(self.actions), )))

    def render(self, info, mode='human'):
        """
        Renders the environment

        :param info: A `dict` of data
        :param mode: A `str` of the available mode
        """
        fig, axes = pyplot.subplots(1, 3, figsize=(10,3), sharey=True, sharex=True)

        axes[0].imshow(info["conf1"])
        axes[0].set_title(f"Datamap roi")

        axes[1].imshow(info["bleached"]["base"][self.datamap.roi])
        axes[1].set_title(f"Bleached datamap")

        axes[2].imshow(info["sted_image"])
        axes[2].set_title(f"Acquired signal (photons)")

        pyplot.show(block=True)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def update_(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _update_datamap(self):
        self.synapse = self.synapse_generator(rotate=True)
        self.datamap = self.microscope_generator.generate_datamap(
            datamap = {
                "whole_datamap" : self.synapse.frame,
                "datamap_pixelsize" : self.microscope_generator.pixelsize
            }
        )

        # Acquire confocal image which sets the current state
        conf_params = self.microscope_generator.generate_params()
        state, _, _ = self.microscope.get_signal_and_bleach(
            self.datamap, self.datamap.pixelsize, **conf_params, bleach=False
        )
        return state

    def _acquire(self, action):

        # Generates imaging parameters
        sted_params = self.microscope_generator.generate_params(
            imaging = {
                name : action[self.actions.index(name)]
                    if name in self.actions else getattr(defaults, name.upper())
                    for name in ["pdt", "p_ex", "p_sted"]
            }
        )
        conf_params = self.microscope_generator.generate_params()

        # Acquire confocal image
        conf1, bleached, _ = self.microscope.get_signal_and_bleach(
            self.datamap, self.datamap.pixelsize, **conf_params, bleach=False
        )

        # Acquire STED image
        sted_image, bleached, _ = self.microscope.get_signal_and_bleach(
            self.datamap, self.datamap.pixelsize, **sted_params, bleach=False
        )

        # Acquire confocal image
        conf2, bleached, _ = self.microscope.get_signal_and_bleach(
            self.datamap, self.datamap.pixelsize, **conf_params, bleach=False
        )

        # automatically bleaches the datamap
        normalized_p_sted = 1 - (sted_params["p_sted"] - defaults.action_spaces["p_sted"]["low"]) / (defaults.action_spaces["p_sted"]["high"] - defaults.action_spaces["p_sted"]["low"])
        conf2 = conf2 * normalized_p_sted
        self.datamap.whole_datamap = self.datamap.whole_datamap * normalized_p_sted

        # foreground on confocal image
        fg_c = get_foreground(conf1)
        # foreground on sted image
        if numpy.any(sted_image):
            fg_s = get_foreground(sted_image)
        else:
            fg_s = numpy.ones_like(fg_c)
        # remove STED foreground points not in confocal foreground, if any
        fg_s *= fg_c

        return sted_image, bleached["base"][self.datamap.roi], conf1, conf2, fg_s, fg_c

    def close(self):
        return None

class STEDEnvWithoutVision(gym.Env):
    """
    Creates a `STEDEnvWithoutVision`

    The `STEDEnvWithoutVision` implements a scan of the entire field of view
    """
    metadata = {'render.modes': ['human']}
    obj_names = ["Resolution", "Bleach", "SNR"]

    def __init__(self, reward_calculator="SumRewardCalculator", actions=["p_sted"]):

        self.synapse_generator = SynapseGenerator(mode="mushroom", seed=None)
        self.microscope_generator = MicroscopeGenerator()
        self.microscope = self.microscope_generator.generate_microscope()

        self.actions = actions
        self.action_space = spaces.Box(
            low=numpy.array([defaults.action_spaces[name]["low"] for name in self.actions]),
            high=numpy.array([defaults.action_spaces[name]["high"] for name in self.actions]),
            shape=(len(self.actions),), dtype=numpy.float32
        )
        self.observation_space = spaces.Box(0, 1, shape=(len(self.obj_names) + len(self.actions),), dtype=numpy.float32)

        self.state = None
        self.initial_count = None
        self.current_step = 0

        objs = OrderedDict({obj_name : obj_dict[obj_name] for obj_name in self.obj_names})
        bounds = OrderedDict({obj_name : bounds_dict[obj_name] for obj_name in self.obj_names})
        scales = OrderedDict({obj_name : scales_dict[obj_name] for obj_name in self.obj_names})
        self.reward_calculator = getattr(rewards, reward_calculator)(objs, bounds=bounds, scales=scales)
        self.mo_reward_calculator = rewards.MORewardCalculator(objs, bounds=bounds, scales=scales)

        self.datamap = None
        self.viewer = None

        self.seed()

    def step(self, action):

        # We manually clip the actions which are out of action space
        action = numpy.clip(action, self.action_space.low, self.action_space.high)

        # Acquire the image
        sted_image, bleached, conf1, conf2, fg_s, fg_c = self._acquire(action)

        reward = self.reward_calculator.evaluate(sted_image, conf1, conf2, fg_s, fg_c)
        mo_objs = self.mo_reward_calculator.evaluate_rescale(sted_image, conf1, conf2, fg_s, fg_c)

        num_killed = (self.initial_count - self.datamap.whole_datamap.sum()) / self.initial_count
        # print(action / defaults.action_spaces["p_sted"]["high"], num_killed)
        done = (self.current_step >= self.spec.max_episode_steps) \
                or (num_killed > 0.90)
        # print(rewards, reward, done)
        observation = conf2[..., numpy.newaxis]
        info = {
            "action" : action,
            "bleached" : bleached,
            "sted_image" : sted_image,
            "conf1" : conf1,
            "conf2" : conf2,
            "fg_c" : fg_c,
            "fg_s" : fg_s,
            "mo_objs" : mo_objs
        }

        self.current_step += 1

        # action  = (action - defaults.action_spaces["p_sted"]["low"]) / (defaults.action_spaces["p_sted"]["high"] - defaults.action_spaces["p_sted"]["low"])

        return numpy.array(mo_objs + action.tolist()), reward, done, info

    def reset(self):
        """
        Resets the environment with a new datamap

        :returns : A `numpy.ndarray` of the molecules
        """
        self.current_step = 0
        state = self._update_datamap()
        self.initial_count = self.datamap.whole_datamap.sum()

        self.state = state[..., numpy.newaxis]
        return numpy.zeros((len(self.obj_names) + len(self.actions), ))

    def render(self, info, mode='human'):
        """
        Renders the environment

        :param info: A `dict` of data
        :param mode: A `str` of the available mode
        """
        fig, axes = pyplot.subplots(1, 3, figsize=(10,3), sharey=True, sharex=True)

        axes[0].imshow(info["conf1"])
        axes[0].set_title(f"Datamap roi")

        axes[1].imshow(info["bleached"]["base"][self.datamap.roi])
        axes[1].set_title(f"Bleached datamap")

        axes[2].imshow(info["sted_image"])
        axes[2].set_title(f"Acquired signal (photons)")

        pyplot.show(block=True)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def update_(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _update_datamap(self):
        self.synapse = self.synapse_generator(rotate=True)
        self.datamap = self.microscope_generator.generate_datamap(
            datamap = {
                "whole_datamap" : self.synapse.frame,
                "datamap_pixelsize" : self.microscope_generator.pixelsize
            }
        )

        # Acquire confocal image which sets the current state
        conf_params = self.microscope_generator.generate_params()
        state, _, _ = self.microscope.get_signal_and_bleach(
            self.datamap, self.datamap.pixelsize, **conf_params, bleach=False
        )
        return state

    def _acquire(self, action):

        # Generates imaging parameters
        sted_params = self.microscope_generator.generate_params(
            imaging = {
                name : action[self.actions.index(name)]
                    if name in self.actions else getattr(defaults, name.upper())
                    for name in ["pdt", "p_ex", "p_sted"]
            }
        )
        conf_params = self.microscope_generator.generate_params()

        # Acquire confocal image
        conf1, bleached, _ = self.microscope.get_signal_and_bleach(
            self.datamap, self.datamap.pixelsize, **conf_params, bleach=False
        )

        # Acquire STED image
        sted_image, bleached, _ = self.microscope.get_signal_and_bleach(
            self.datamap, self.datamap.pixelsize, **sted_params, bleach=False
        )

        # Acquire confocal image
        conf2, bleached, _ = self.microscope.get_signal_and_bleach(
            self.datamap, self.datamap.pixelsize, **conf_params, bleach=False
        )

        # automatically bleaches the datamap
        normalized_p_sted = 1 - (sted_params["p_sted"] - defaults.action_spaces["p_sted"]["low"]) / (defaults.action_spaces["p_sted"]["high"] - defaults.action_spaces["p_sted"]["low"])
        conf2 = conf2 * normalized_p_sted
        self.datamap.whole_datamap = self.datamap.whole_datamap * normalized_p_sted

        # foreground on confocal image
        fg_c = get_foreground(conf1)
        # foreground on sted image
        if numpy.any(sted_image):
            fg_s = get_foreground(sted_image)
        else:
            fg_s = numpy.ones_like(fg_c)
        # remove STED foreground points not in confocal foreground, if any
        fg_s *= fg_c

        return sted_image, bleached["base"][self.datamap.roi], conf1, conf2, fg_s, fg_c

    def close(self):
        return None

class STEDEnvWithDelayedReward(STEDEnv):
    """
    Creates a `STEDEnvWithDelayedReward`
    """
    metadata = {'render.modes': ['human']}
    obj_names = ["Resolution", "Bleach", "SNR"]

    def __init__(self, reward_calculator="SumRewardCalculator", actions=["p_sted"]):
        super(STEDEnvWithDelayedReward, self).__init__(
            reward_calculator=reward_calculator,
            actions = actions
        )

        self.episode_memory = []

    def step(self, action):
        # We manually clip the actions which are out of action space
        action = numpy.clip(action, self.action_space.low, self.action_space.high)

        # Acquire the image
        sted_image, bleached, conf1, conf2, fg_s, fg_c = self._acquire(action)

        reward = self.reward_calculator.evaluate(sted_image, conf1, conf2, fg_s, fg_c)
        mo_objs = self.mo_reward_calculator.evaluate(sted_image, conf1, conf2, fg_s, fg_c)

        num_killed = (self.initial_count - self.datamap.whole_datamap.sum()) / self.initial_count
        # print(action / defaults.action_spaces["p_sted"]["high"], num_killed)
        done = (self.current_step >= self.spec.max_episode_steps) \
                or (num_killed > 0.90)
        # print(rewards, reward, done)
        observation = conf2[..., numpy.newaxis]
        info = {
            "action" : action,
            "bleached" : bleached,
            "sted_image" : sted_image,
            "conf1" : conf1,
            "conf2" : conf2,
            "fg_c" : fg_c,
            "fg_s" : fg_s,
            "mo_objs" : mo_objs
        }
        self.current_step += 1

        self.episode_memory.append(reward)
        if done:
            reward = numpy.array(self.episode_memory)
        else:
            reward = 0

        return (observation, numpy.array(mo_objs + action.tolist())), reward, done, info

    def reset(self):
        """
        Resets the environment with a new datamap

        :returns : A `numpy.ndarray` of the molecules
        """
        self.current_step = 0
        self.episode_memory = []
        state = self._update_datamap()
        self.initial_count = self.datamap.whole_datamap.sum()

        self.state = state[..., numpy.newaxis]
        return (self.state, numpy.zeros((len(self.obj_names) + len(self.actions), )))

if __name__ == "__main__":

    from pysted.utils import mse_calculator

    env = STEDEnv(reward_calculator="SumRewardCalculator", actions=["p_sted"])
    images = []
    for i in numpy.linspace(env.action_space.low[0], env.action_space.high[0], 10):
        obs = env.reset()
        images.append(obs[0])
        # obs, reward, done, info = env.step(env.action_space.sample())
        obs, reward, done, info = env.step(numpy.array(i))
        print(reward, info["rewards"], info["action"])
