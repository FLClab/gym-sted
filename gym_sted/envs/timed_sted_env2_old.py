
import gym
import numpy
import random
import os
import queue

import pysted.base
from gym import error, spaces, utils
from gym.utils import seeding
from matplotlib import pyplot
from collections import OrderedDict

import gym_sted
from gym_sted import rewards, defaults
from gym_sted.utils import SynapseGenerator2, MicroscopeGenerator, RecordingQueue, get_foreground
from gym_sted.rewards import objectives_timed2, rewards_timed2
from gym_sted.prefnet import PreferenceArticulator

# I will copy the values straight from Anthony's ranking_sted_env, need to think about what values
# I actually want to use

obj_dict = {
    "SNR" : objectives_timed2.Signal_Ratio(75),
    "Bleach" : objectives_timed2.Bleach(),
    "Resolution" : objectives_timed2.Resolution(pixelsize=20e-9),
    "NbNanodomains" : objectives_timed2.NumberNanodomains()
}
bounds_dict = {
    "SNR" : {"min" : 0.20, "max" : numpy.inf},
    "Bleach" : {"min" : -numpy.inf, "max" : 0.5},
    "Resolution" : {"min" : 0, "max" : 100},
    "NbNanodomains" : {"min" : 0, "max" : numpy.inf}
}
scales_dict = {
    "SNR" : {"min" : 0, "max" : 1},
    "Bleach" : {"min" : 0, "max" : 1},
    "Resolution" : {"min" : 40, "max" : 180},
    "NbNanodomains" : {"min" : 0, "max" : 1}   # ???
}
action_spaces = {
    # changed p_sted low to 0 as I want to 0. as I want to take confocals if the flash is not yet happening
    "p_sted" : {"low" : 0., "high" : 5.0e-3},
    "p_ex" : {"low" : 0., "high" : 5.0e-6},   # jveux tu lui laisser prendre un p_ex = 0 ? ferait la wait action...
    "pdt" : {"low" : 10.0e-6, "high" : 150.0e-6},
}


class timedExpSTEDEnv2old(gym.Env):
    """
    Creates a 'STEDEnv'

    Action space
        The action space here is a selection of the p_ex, p_sted and pdt values (?)
    """
    metadata = {'render.modes': ['human']}
    obj_names = ["Resolution", "Bleach", "SNR", "NbNanodomains"]

    def __init__(self, time_quantum_us=1, exp_time_us=2000000, actions=["p_sted"],
                 reward_calculator="NanodomainsRewardCalculator"):
        # self.synapse_generator = SynapseGenerator2(mode="mushroom", n_nanodomains=7, n_molecs_in_domain=100, seed=42)
        self.synapse_generator = SynapseGenerator2(mode="mushroom", n_nanodomains=7, n_molecs_in_domain=5, seed=42)
        self.microscope_generator = MicroscopeGenerator()
        self.microscope = self.microscope_generator.generate_microscope()

        self.actions = actions
        self.action_space = spaces.Box(
            low=numpy.array([action_spaces[name]["low"] for name in self.actions]),
            high=numpy.array([action_spaces[name]["high"] for name in self.actions]),
            shape=(len(self.actions),),
            dtype=numpy.float32
        )

        self.dmap_shape = (64, 64)
        self.q_length = 4
        self.max_episode_steps = int(numpy.ceil(exp_time_us /
                                                (self.dmap_shape[0] * self.dmap_shape[1] * action_spaces["pdt"]["low"] * 1e6)))
        # since this is a temporal experiment, I think it would be relevant to have the last 4 acquisitions as the
        # observation of the state. Need to figure out how to do a first in first out thing for this
        # self.observation_space = spaces.Box(0, 2 ** 16, shape=(self.dmap_shape[0], self.dmap_shape[1], self.q_length),
        #                                     dtype=numpy.uint16)
        self.observation_space = spaces.Tuple((
            spaces.Box(0, 2 ** 16, shape=(self.dmap_shape[0], self.dmap_shape[1], self.q_length),
                       dtype=numpy.uint16),
            spaces.Box(0, scales_dict["Resolution"]["max"], shape=(3,), dtype=numpy.float32)
        ))


        self.state = None
        self.initial_count = None

        self.cummulated_rewards = {
            "rewards": [],
            "reward": []
        }

        objs = OrderedDict({obj_name: obj_dict[obj_name] for obj_name in self.obj_names})
        bounds = OrderedDict({obj_name: bounds_dict[obj_name] for obj_name in self.obj_names})
        scales = OrderedDict({obj_name: scales_dict[obj_name] for obj_name in self.obj_names})
        self.reward_calculator = getattr(rewards_timed2, reward_calculator)(objs, bounds=bounds, scales=scales)
        self._reward_calculator = rewards_timed2.MORewardCalculator(objs, bounds=bounds, scales=scales)

        self.temporal_datamap = None
        self.viewer = None

        self.time_quantum_us = time_quantum_us
        self.exp_time_us = exp_time_us

        self.clock = None
        self.temporal_experiment = None

        self.seed()

    def step(self, action):
        """
        hmmmm
        faque là il faudrait que je split ça en 2 actions, une qui serait du "monitoring" pour laquelle je ne compute
        pas le nb de nanodomaines et la reward associée et une action "d'acquisition" pour laquelle je compute ça.
        Dans les 2 cas, je veux envoyer le signal de SNR, Bleach, Resolution au NN, mais je veu juste obtenir une
        reward quand je fais un guess sur le nombre de nanodomaines.
        hmmmm
        dont d'abord je devrais regarder comment envoyer le signal des 3 objs au neural net and shit
        """
        # Define a sted power under which we are in 'monitoring' mode, over which we are in 'detection' mode
        p_sted_threshold = 5e-4

        action = numpy.clip(action, self.action_space.low, self.action_space.high)

        # là il faut que je regarde si la puissance sted est under/over le threshold pour décider comment retourner la
        # reward. Il faut aussi que je regarde l'affaire avec le pdt :)

        # Generates imaging parameters
        sted_params = self.microscope_generator.generate_params(
            imaging={
                name: action[self.actions.index(name)] * numpy.ones(self.dmap_shape) if name == "pdt"
                else action[self.actions.index(name)]
                if name in self.actions else getattr(defaults, name.upper())
                for name in ["pdt", "p_ex", "p_sted"]
            }
        )
        if sted_params["pdt"][0, 0] * 1e6 + self.temporal_experiment.clock.current_time >= self.exp_time_us:
            # case where the agent doesnt have the time to image a single pixel given the time left in the experiment
            # and the selected pdt

            # set the sted_image, conf1, fg_c and fg_s to a bunch of zeros,
            sted_image = numpy.zeros(self.dmap_shape)
            conf1 = numpy.zeros(self.dmap_shape)
            fg_c = numpy.zeros(self.dmap_shape)
            fg_s = numpy.zeros(self.dmap_shape)
            n_molecs_init = self.temporal_datamap.base_datamap.sum(),
            n_molecs_post = self.temporal_datamap.base_datamap.sum()

            # this is the scalarized reward
            reward = self.reward_calculator.evaluate(sted_image, conf1, fg_s, fg_c, n_molecs_init, n_molecs_post,
                                                     self.temporal_datamap)
            # this is the vector of individual rewards for individual objectives
            rewards = self._reward_calculator.evaluate(sted_image, conf1, fg_s, fg_c, n_molecs_init, n_molecs_post,
                                                       self.temporal_datamap)

            done = True
            info = {
                "action": action,
                "bleached": None,  # not sure what to do here, will this cause problems? hope not :)
                "sted_image": sted_image,
                "conf1": conf1,
                "fg_c": fg_c,
                "fg_s": fg_s,
                "rewards": rewards,
                "pdt": action[self.actions.index("pdt")],
                "p_ex": action[self.actions.index("p_ex")],
                "p_sted": action[self.actions.index("p_sted")]
            }

            # faut que j'update mon state avec ma plus récente acq :)
            self.state.enqueue(sted_image)
            observation = numpy.transpose(self.state.to_array(), (1, 2, 0))
            objective_vals = numpy.array([rewards["SNR"], rewards["Resolution"], rewards["Bleach"]])

            # ~!* RETURN DLA SCRAP ICITTE *!~
            return [observation, objective_vals], reward, done, info

        else:
            # case where the agent can image at least one pixel given the time left in the exp and selected pdt

            # FAUT QUE JE SPLIT ÇA EN 2 CAS, UN POUR LE MONITORING ET UN POUR LE DETECTION

            conf_params = self.microscope_generator.generate_params()

            # Acquire confocal image
            conf1, bleached, _ = self.microscope.get_signal_and_bleach(
                self.temporal_datamap, self.temporal_datamap.pixelsize, **conf_params, bleach=False
            )

            n_molecs_init = self.temporal_datamap.base_datamap.sum()

            # Acquire a STED image (with all the time wizardy bullshit)
            sted_image, bleached = self.temporal_experiment.play_action(**sted_params)

            n_molecs_post = self.temporal_datamap.base_datamap.sum()

            # foreground on confocal image
            fg_c = get_foreground(conf1)
            # foreground on sted image
            if numpy.any(sted_image):
                fg_s = get_foreground(sted_image)
            else:
                fg_s = numpy.ones_like(fg_c)
            # remove STED foreground points not in confocal foreground, if any
            fg_s *= fg_c

            # this is the vector of individual rewards for individual objectives
            rewards = self._reward_calculator.evaluate(sted_image, conf1, fg_s, fg_c, n_molecs_init, n_molecs_post,
                                                       self.temporal_datamap)

            # compute rewards differently for monitoring or detection mode
            if sted_params["p_sted"] < p_sted_threshold:   # monitoring
                reward = 0
                rewards["NbNanodomains"] = 0
            else:   # detection
                # this is the scalarized reward
                reward = self.reward_calculator.evaluate(sted_image, conf1, fg_s, fg_c, n_molecs_init, n_molecs_post,
                                                         self.temporal_datamap)

            done = self.temporal_experiment.clock.current_time >= self.exp_time_us

            info = {
                "action": action,
                "bleached": bleached,
                "sted_image": sted_image,
                "conf1": conf1,
                "fg_c": fg_c,
                "fg_s": fg_s,
                "rewards": rewards,
                "pdt": action[self.actions.index("pdt")],
                "p_ex": action[self.actions.index("p_ex")],
                "p_sted": action[self.actions.index("p_sted")]
            }

            # faut que j'update mon state avec ma plus récente acq :)
            self.state.enqueue(sted_image)
            observation = numpy.transpose(self.state.to_array(), (1, 2, 0))
            objective_vals = numpy.array([rewards["SNR"], rewards["Resolution"], rewards["Bleach"]])

            # ~!* RETURN DLA SCRAP ICITTE *!~
            return [observation, objective_vals], reward, done, info

    def reset(self):
        synapse = self.synapse_generator.generate()

        self.temporal_datamap = self.microscope_generator.generate_temporal_datamap(
            temporal_datamap = {
                "whole_datamap" : synapse.frame,
                "datamap_pixelsize" : self.microscope_generator.pixelsize,
                "synapse_obj": synapse
            },
            decay_time_us=self.exp_time_us,
            n_decay_steps=20
        )
        n_molecs_init = self.temporal_datamap.base_datamap.sum()
        conf_params = self.microscope_generator.generate_params()
        first_acq, _, _ = self.microscope.get_signal_and_bleach(
            self.temporal_datamap, self.temporal_datamap.pixelsize, **conf_params, bleach=False
        )
        n_molecs_post = self.temporal_datamap.base_datamap.sum()
        # for the state, fill a queue with the first observation,
        # If I only add it once to the Q the head will point to an array of zeros until the Q is filled,
        # which is not what I want
        start_data = []
        for i in range(self.q_length):
            start_data.append(first_acq)
        start_data = numpy.array(start_data)
        self.state = RecordingQueue(start_data, maxlen=self.q_length, num_sensors=self.dmap_shape)
        # là mon state c'est un Q object, est-ce que c'est good? pour les passer à mes nn? idk?

        self.clock = pysted.base.Clock(self.time_quantum_us)
        self.temporal_experiment = pysted.base.TemporalExperiment(self.clock, self.microscope, self.temporal_datamap,
                                                                  self.exp_time_us, bleach=True,
                                                                  bleach_mode="proportional")

        # foreground on confocal image
        fg_c = get_foreground(first_acq)
        # foreground on sted image
        if numpy.any(first_acq):
            fg_s = get_foreground(first_acq)
        else:
            fg_s = numpy.ones_like(fg_c)
        # remove STED foreground points not in confocal foreground, if any
        fg_s *= fg_c

        rewards = self._reward_calculator.evaluate(first_acq, first_acq, fg_s, fg_c, n_molecs_init, n_molecs_post,
                                                   self.temporal_datamap)

        # faut aussi que je retourne le vecteur avec [SNR, Resolution, Bleach] ... how?
        # caluler les rewards avec MORewardsCalculator et utiliser ça I guess? ou juste retourne [0, 0, 0] ?
        objective_vals = numpy.array([rewards["SNR"], rewards["Resolution"], rewards["Bleach"]])
        return [numpy.transpose(self.state.to_array(), (1, 2, 0)), objective_vals]

    def render(self, info, mode='human'):
        """
        unsure when this will be used
        Renders the environment

        :param info: A `dict` of data
        :param mode: A `str` of the available mode
        """
        fig, axes = pyplot.subplots(1, 3, figsize=(10, 3), sharey=True, sharex=True)

        axes[0].imshow(info["conf1"])
        axes[0].set_title(f"Datamap roi")

        axes[1].imshow(info["bleached"]["base"][self.temporal_datamap.roi])
        axes[1].set_title(f"Bleached datamap")

        axes[2].imshow(info["sted_image"])
        axes[2].set_title(f"Acquired signal (photons)")

        pyplot.show(block=True)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def update_(self, **kwargs):
        # unsure when this will be used
        for key, value in kwargs.items():
            setattr(self, key, value)

    def close(self):
        return None


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    # ces params me donned une Resolution de -0.5 ???
    # env = timedExpSTEDEnv(actions=["pdt", "p_ex", "p_sted"])
    # state = env.reset()
    # env.clock.current_time = 2000000 - 150
    # obs, reward, done, info = env.step([10, 10, 0.00005])
    # print(f"info = {info}")
    # print(f"reward = {reward}")

    env = timedExpSTEDEnv2old(actions=["pdt", "p_ex", "p_sted"])
    state = env.reset()
    env.clock.current_time = 0
    obs, reward, done, info = env.step([10, 10, 0.5])
    print(f"info = {info}")
    print(f"reward = {reward}")
    plt.imshow(info["sted_image"])
    plt.show()