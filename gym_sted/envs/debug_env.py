import gymnasium as gym
import numpy
import random

import pysted.base
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding
from matplotlib import pyplot
from collections import OrderedDict

import gym_sted
from gym_sted import rewards, defaults
from gym_sted.utils import SynapseGenerator, MicroscopeGenerator, get_foreground, BleachSampler
from gym_sted.rewards import objectives
from gym_sted.prefnet import PreferenceArticulator

obj_dict = {
    "SNR" : objectives.Signal_Ratio(75),
    "Bleach" : objectives.Bleach(),
    "Resolution" : objectives.Resolution(pixelsize=20e-9),
    "Squirrel" : objectives.Squirrel(),
    "NbNanodomains" : objectives.NumberNanodomains()
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
    "NbNanodomains" : {"min" : 0, "max" : 1}
}
action_spaces = {
    "p_sted" : {"low" : 0., "high" : 5.0e-3},
    "p_ex" : {"low" : 0.8e-6, "high" : 5.0e-6},
    "pdt" : {"low" : 10.0e-6, "high" : 150.0e-6},
}

class DebugBleachSTEDEnv(gym.Env):

    obj_names = ["Bleach"]

    def __init__(self, reward_calculator="SumRewardCalculator", actions=["p_sted", "p_ex", "pdt"]):

        self.synapse_generator = SynapseGenerator(mode="mushroom", seed=42)
        self.microscope_generator = MicroscopeGenerator()
        self.microscope = self.microscope_generator.generate_microscope()

        self.actions = actions
        self.action_space = spaces.Box(
            low=numpy.array([action_spaces[name]["low"] for name in self.actions]),
            high=numpy.array([action_spaces[name]["high"] for name in self.actions]),
            shape=(len(self.actions),), dtype=numpy.float32
        )
        self.observation_space = spaces.Box(0, 2**16, shape=(20, 20, 1), dtype=numpy.uint16)

        self.state = None
        self.initial_count = None

        objs = OrderedDict({obj_name : obj_dict[obj_name] for obj_name in self.obj_names})
        bounds = OrderedDict({obj_name : bounds_dict[obj_name] for obj_name in self.obj_names})
        scales = OrderedDict({obj_name : scales_dict[obj_name] for obj_name in self.obj_names})
        self.reward_calculator = getattr(rewards, reward_calculator)(objs, bounds=bounds, scales=scales)
        self._reward_calculator = rewards.MORewardCalculator(objs, bounds=bounds, scales=scales)

        self.datamap = None
        self.viewer = None

        # self.clear_statistics()

        self.seed()

    def step(self, action):

        # We manually rescale and clip the actions which are out of action space
        action = numpy.clip(action, self.action_space.low, self.action_space.high)

        # self.statistics["mean-action"].append(action)

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
            self.datamap, self.datamap.pixelsize, **sted_params, bleach=True
        )

        # Acquire confocal image
        conf2, bleached, _ = self.microscope.get_signal_and_bleach(
            self.datamap, self.datamap.pixelsize, **conf_params, bleach=False
        )

        # foreground on confocal image
        fg_c = get_foreground(conf1)
        # foreground on sted image
        if numpy.any(sted_image):
            fg_s = get_foreground(sted_image)
        else:
            fg_s = numpy.ones_like(fg_c)
        # remove STED foreground points not in confocal foreground, if any
        fg_s *= fg_c

        # Only calculates the bleach
        reward = self.reward_calculator.evaluate(sted_image, conf1, conf2, fg_s, fg_c)
        rewards = self._reward_calculator.evaluate(sted_image, conf1, conf2, fg_s, fg_c)

        done = conf2.sum() < 1.
        # print("EHY", done, reward)
        observation = conf2[..., numpy.newaxis]
        info = {
            "action" : action,
            "bleached" : bleached,
            "sted_image" : sted_image,
            "conf1" : conf1,
            "conf2" : conf2,
            "fg_c" : fg_c,
            "fg_s" : fg_s,
            "rewards" : rewards
        }

        return observation, reward, done, info

    def reset(self, seed):
        """
        Resets the environment with a new datamap
        :returns : A `numpy.ndarray` of the molecules
        """
        super().reset(seed=seed)
        molecules_disposition = numpy.zeros((20, 20))
        molecules_disposition[5, 5] = 10
        self.datamap = self.microscope_generator.generate_datamap(
            datamap = {
                "whole_datamap" : molecules_disposition,
                "datamap_pixelsize" : self.microscope_generator.pixelsize
            }
        )

        # Acquire confocal image which sets the current state
        conf_params = self.microscope_generator.generate_params()
        self.state, _, _ = self.microscope.get_signal_and_bleach(
            self.datamap, self.datamap.pixelsize, **conf_params, bleach=False
        )

        self.initial_count = molecules_disposition.sum()
        return self.state[..., numpy.newaxis]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # def get_statistics(self):
    #     # return [("mean-action", [])]
    #     return [tuple((key, numpy.mean(value, axis=0))) if value else tuple((key, None)) for key, value in self.statistics.items()]
    #
    # def clear_statistics(self):
    #     self.statistics = {
    #         "mean-action" : []
    #     }
    #     return

    def close(self):
        return None

class DebugResolutionSNRSTEDEnv(gym.Env):

    obj_names = ["Resolution", "SNR"]

    def __init__(self, reward_calculator="MORewardCalculator", actions=["p_sted"]):

        self.synapse_generator = SynapseGenerator(mode="mushroom", seed=42)
        self.microscope_generator = MicroscopeGenerator()
        self.microscope = self.microscope_generator.generate_microscope()

        self.actions = actions
        self.action_space = spaces.Box(
            low=numpy.array([action_spaces[name]["low"] for name in self.actions]),
            high=numpy.array([action_spaces[name]["high"] for name in self.actions]),
            shape=(len(self.actions),), dtype=numpy.float32
        )
        self.observation_space = spaces.Box(0, 2**16, shape=(64, 64, 1), dtype=numpy.uint16)

        self.state = None
        self.initial_count = None

        objs = OrderedDict({obj_name : obj_dict[obj_name] for obj_name in self.obj_names})
        bounds = OrderedDict({obj_name : bounds_dict[obj_name] for obj_name in self.obj_names})
        scales = OrderedDict({obj_name : scales_dict[obj_name] for obj_name in self.obj_names})
        self.reward_calculator = getattr(rewards, reward_calculator)(objs, bounds=bounds, scales=scales)
        self._reward_calculator = rewards.MORewardCalculator(objs, bounds=bounds, scales=scales)

        self.datamap = None
        self.viewer = None

        self.seed()

    def step(self, action):

        # We manually rescale and clip the actions which are out of action space
        action = numpy.clip(action, self.action_space.low, self.action_space.high)

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

        # foreground on confocal image
        fg_c = get_foreground(conf1)
        # foreground on sted image
        if numpy.any(sted_image):
            fg_s = get_foreground(sted_image)
        else:
            fg_s = numpy.ones_like(fg_c)
        # remove STED foreground points not in confocal foreground, if any
        fg_s *= fg_c

        reward = self.reward_calculator.evaluate(sted_image, conf1, conf2, fg_s, fg_c)
        reward = (1 - (reward[0] - 40) / (250 - 40)) + reward[1]
        reward = reward.item()
        rewards = self._reward_calculator.evaluate(sted_image, conf1, conf2, fg_s, fg_c)

        done = numpy.all(action == self.action_space.high)

        observation = conf2[..., numpy.newaxis]
        info = {
            "action" : action,
            "bleached" : bleached,
            "sted_image" : sted_image,
            "conf1" : conf1,
            "conf2" : conf2,
            "fg_c" : fg_c,
            "fg_s" : fg_s,
            "rewards" : rewards
        }

        return observation, reward, done, info

    def reset(self, seed=None):
        """
        Resets the environment with a new datamap
        :returns : A `numpy.ndarray` of the molecules
        """
        super().reset(seed=seed)
        molecules_disposition = self.synapse_generator()
        self.datamap = self.microscope_generator.generate_datamap(
            datamap = {
                "whole_datamap" : molecules_disposition,
                "datamap_pixelsize" : self.microscope_generator.pixelsize
            }
        )

        # Acquire confocal image which sets the current state
        conf_params = self.microscope_generator.generate_params()
        self.state, _, _ = self.microscope.get_signal_and_bleach(
            self.datamap, self.datamap.pixelsize, **conf_params, bleach=False
        )

        self.initial_count = molecules_disposition.sum()
        return self.state[..., numpy.newaxis]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        return None

class DebugBleachSTEDTimedEnv(gym.Env):

    obj_names = ["Resolution", "Bleach", "SNR"]

    def __init__(self, time_quantum_us=1, exp_time_us=500000, reward_calculator="MORewardCalculator"):

        self.synapse_generator = SynapseGenerator(mode="mushroom", seed=42)
        self.microscope_generator = MicroscopeGenerator()
        self.microscope = self.microscope_generator.generate_microscope()

        self.action_space = spaces.Box(low=0., high=5e-3, shape=(1,), dtype=numpy.float32)
        self.observation_space = spaces.Box(0, 2 ** 16, shape=(1, 20, 20), dtype=numpy.uint16)

        self.state = None
        self.initial_count = None

        objs = OrderedDict({obj_name: obj_dict[obj_name] for obj_name in self.obj_names})
        bounds = OrderedDict({obj_name: bounds_dict[obj_name] for obj_name in self.obj_names})
        # self.reward_calculator = BoundedRewardCalculator(objs, bounds)
        scales = OrderedDict({obj_name: scales_dict[obj_name] for obj_name in self.obj_names})
        self.reward_calculator = getattr(rewards, reward_calculator)(objs, bounds=bounds, scales=scales)
        # self._reward_calculator = RewardCalculator(objs)

        self.temporal_datamap = None
        self.viewer = None

        # Do I need to init other stuff since this is a timed exp ?
        # maybe I need an exp_time, time_quantum_us, clock ?
        # These 2 vals will be the same for every exp (episode) and do not change during an exp so no need to be reset
        self.time_quantum_us = time_quantum_us
        self.exp_time_us = exp_time_us

        self.clock = None
        self.temporal_experiment = None

        self.seed()

    def step(self, action):

        # We manually rescale and clip the actions which are out of action space
        m, M = -1, 1
        action = (action - m) / (M - m)
        action = action * (self.action_space.high - self.action_space.low) + self.action_space.low
        action = numpy.clip(action, self.action_space.low, self.action_space.high)

        # Generates imaging parameters
        sted_params = self.microscope_generator.generate_params(
            imaging={
                "pdt": numpy.ones(self.temporal_datamap.whole_datamap[self.temporal_datamap.roi].shape) * 100.0e-6,
                "p_ex": 2.0e-6,
                "p_sted": action[0]   # for now the agent only controls the STED power
            }
        )

        """
        where were we
        for the first version of my debugging exp, I will do:
        acquire an 'instant' confocal
            - instant meaning it doesn't affect the time of the exp / get interrupted by flash updates or w/e
        acquire the STED
            - with all the time management wizardry
        acquire an 'instant' confocal
        Then compute the difference between to two confocals to see how much I bleached to give the reward
        """
        conf_params = self.microscope_generator.generate_params()

        # Acquire an 'instant' confocal image
        # conf1, bleached, _ = self.microscope.get_signal_and_bleach(
        #     self.temporal_datamap, self.temporal_datamap.pixelsize, **conf_params, bleach=False
        # )
        # Instead of using confocals I will directly use the number of molecules in the "base"
        n_molecs_init = self.temporal_datamap.base_datamap.sum()

        # Acquire a STED image (with all the time wizardy bullshit)
        sted_image, bleached = self.temporal_experiment.play_action(**sted_params)

        # Acquire an 'instant' confocal image
        # conf2, bleached, _ = self.microscope.get_signal_and_bleach(
        #     self.temporal_datamap, self.temporal_datamap.pixelsize, **conf_params, bleach=False
        # )
        # Instead of using confocals I will directly use the number of molecules in the "base"
        n_molecs_post = self.temporal_datamap.base_datamap.sum()


        # do stuff to compute the rewards and stuff here
        reward = (n_molecs_init - n_molecs_post) / n_molecs_init

        # done when either everything bleached or the clock_time is greater than the exp time
        done = (n_molecs_post < 1) or (self.temporal_experiment.clock.current_time >= self.exp_time_us)

        observation = sted_image[numpy.newaxis, ...] / 1024.0   # normalize to ensure good NN learning
        # input(f"OBSERVATION SHAPE = {observation.shape}")
        info = {
            "bleached": bleached,
            "sted_image": sted_image,
            "n molecules init": n_molecs_init,
            "n molecules post": n_molecs_post,
            "sted_power": action[0]
        }

        # return something eventually :)
        return observation, reward, done, info

    def reset(self, seed=None):
        """
        Resets the environment with a new datamap
        :returns: A `TemporalDatmap` object containing the evolution of the datamap as the flash occurs
        """
        super().reset(seed=seed)
        molecules_disposition = numpy.ones((20, 20)) * 5   # create a filled square dmap for debugging purpouses
        self.temporal_datamap = self.microscope_generator.generate_temporal_datamap(
            temporal_datamap = {
                "whole_datamap" : molecules_disposition,
                "datamap_pixelsize" : self.microscope_generator.pixelsize
            }
        )
        # a temporal_datamap has been created with its flash_tstack, so we are ready to start an experiment loop?
        # what else am I missing?
        # tous les ptits gars qui étaient à None dans le init doivent avoir une valeur mtn si je comprends bien
        # pas certain de comment je veux gérer le init state vu que prendre une img prend du temps, je veux tu
        # assumer qu'une confocale est fait et qu'ensuite l'exp est lancée? ou qu'une exp commence toujours avec
        # une confocal avec des params fix que l'agent ne choisit pas donc chaque exp a toujours une qté de temps
        # bouffée au début? hmm cas 1 est plus simple for sure mais idk :) je vais assumer cas 1 for now :)
        # Acquire confocal image which sets the current state
        conf_params = self.microscope_generator.generate_params()
        self.state, _, _ = self.microscope.get_signal_and_bleach(
            self.temporal_datamap, self.temporal_datamap.pixelsize, **conf_params, bleach=False
        )

        # comment est-ce que je veux calculer mon bleaching dans le cas où la datamap flash dans le temps?
        self.initial_count = self.temporal_datamap.base_datamap.sum()

        # reset the other params specific to the timed exp paradigm
        self.clock = pysted.base.Clock(self.time_quantum_us)
        self.temporal_experiment = pysted.base.TemporalExperiment(self.clock, self.microscope, self.temporal_datamap,
                                                                  self.exp_time_us, bleach=True)

        # that's it ? ... ???
        return self.state[numpy.newaxis, ...] / 1024.0   # normalize to ensure good NN learning

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        return None

class DebugRankSTEDRecurrentMultiObjectivesEnv(gym.Env):
    """
    Creates a `rankSTEDRecurrentMultiObjectivesEnv`

    Action space
        The action space is a tuple, where
        1. The imaging parameter selection
        2. The action to choose from
            {0: Acquire image, 1: Ask for expert knowledge, 2: Final parameters}

    Observation space
        The observation space is a tuple, where
        1. The current confocal image
        2. A vector containing the current articulation, the selected actions, the obtained objectives

    """
    metadata = {'render.modes': ['human']}
    obj_names = ["Resolution", "Bleach", "SNR"]

    def __init__(self, bleach_sampling="constant", actions=["p_sted"],
                    max_num_requests=1, max_episode_steps=10, select_final=False,
                    scale_rank_reward=False, scale_nanodomain_reward=1.):

        self.actions = actions
        self.select_final = select_final
        self.scale_nanodomain_reward = scale_nanodomain_reward

        if self.select_final:
            self.action_space = spaces.Box(
                low=numpy.array([action_spaces[name]["low"] for name in self.actions] + [0]),
                high=numpy.array([action_spaces[name]["high"] for name in self.actions] + [2 + 1]),
                dtype=numpy.float32
            )
        else:
            self.action_space = spaces.Box(
                low=numpy.array([action_spaces[name]["low"] for name in self.actions] + [0]),
                high=numpy.array([action_spaces[name]["high"] for name in self.actions] + [1 + 1]),
                dtype=numpy.float32
            )

        self.observation_space = spaces.Tuple((
            spaces.Box(0, 2**16, shape=(64, 64, 1), dtype=numpy.uint16),
            spaces.Box(
                0, 1, shape=(max_episode_steps + len(self.obj_names) + len(self.actions),),
                dtype=numpy.float32
            ) # Articulation, shape is given by articulation vector, objectives, actions
        ))

        self.state = None
        self.initial_count = None
        self.synapse = None
        self.current_step = 0
        self.max_num_requests = max_num_requests
        self.num_request_left = max_num_requests
        self.bleach_sampling = bleach_sampling
        self.scale_rank_reward = scale_rank_reward
        self.current_articulation = -1
        self.episode_memory = {
            "actions" : [],
            "mo_objs" : [],
            "reward" : [],
        }

        self.datamap = None
        self.viewer = None

        self.seed()

        self.synapse_generator = SynapseGenerator(mode="rand", seed=None)
        self.microscope_generator = MicroscopeGenerator()
        self.microscope = self.microscope_generator.generate_microscope()
        self.bleach_sampler = BleachSampler(mode=self.bleach_sampling)

        objs = OrderedDict({obj_name : obj_dict[obj_name] for obj_name in self.obj_names})
        bounds = OrderedDict({obj_name : bounds_dict[obj_name] for obj_name in self.obj_names})
        scales = OrderedDict({obj_name : scales_dict[obj_name] for obj_name in self.obj_names})
        self.mo_reward_calculator = rewards.MORewardCalculator(objs, bounds=bounds, scales=scales)
        self.nb_reward_calculator = rewards.NanodomainsRewardCalculator(
            {"NbNanodomains" : obj_dict["NbNanodomains"]},
            bounds={"NbNanodomains" : bounds_dict["NbNanodomains"]},
            scales={"NbNanodomains" : scales_dict["NbNanodomains"]}
        )

        # Loads preference articulation model
        self.preference_articulation = PreferenceArticulator()

    def get_memory_usage(self):
        import psutil, os
        process = psutil.Process(os.getpid())
        print("[----] Current memory usage : {:0.4f}%".format(process.memory_percent()))

    def step(self, action):

        self.get_memory_usage()

        # Action is an array of size self.actions and main_action
        # main action should be in the [0, 1, 2]
        # We manually clip the actions which are out of action space
        action = numpy.nan_to_num(action)
        action = numpy.clip(action, self.action_space.low, self.action_space.high)
        if self.select_final:
            imaging_action, main_action = action[:len(self.actions)], min(int(action[-1]), 2)
        else:
            imaging_action, main_action = action[:len(self.actions)], min(int(action[-1]), 1)

        # On the last step of the environment we enforce the final decision
        if self.current_step >= self.spec.max_episode_steps - 1:
            main_action = 2

        if (main_action == 0) or (main_action == 1 and self.num_request_left <= 0):
            # Acquire an image with the given parameters
            sted_image, bleached, conf1, conf2, fg_s, fg_c = self._acquire(imaging_action)
            mo_objs = self.mo_reward_calculator.evaluate(sted_image, conf1, conf2, fg_s, fg_c)

            # Reward is proportionnal to the ranked position of the last image
            articulation, sorted_indices = self.preference_articulation.articulate(
                self.episode_memory["mo_objs"] + [mo_objs]
            )
            index = numpy.argmax(sorted_indices).item()
            # Reward is given by the position in the sorting
            reward = (index + 1) / len(sorted_indices)

            done = False

        elif main_action == 1:
            # Asking for preference
            sted_image = numpy.zeros(self.observation_space[0].shape[:-1])
            bleached = numpy.zeros(self.observation_space[0].shape[:-1])
            conf1 = numpy.zeros(self.observation_space[0].shape[:-1])
            conf2 = numpy.zeros(self.observation_space[0].shape[:-1])
            fg_s = numpy.zeros(self.observation_space[0].shape[:-1])
            fg_c = numpy.zeros(self.observation_space[0].shape[:-1])

            reward = 1
            mo_objs = [scales_dict[obj_name]["max"] if obj_name != "SNR" else scales_dict[obj_name]["min"] for obj_name in self.obj_names]
            self.num_request_left -= 1

            if len(self.episode_memory["mo_objs"]) > 0:
                self.current_articulation, _ = self.preference_articulation.articulate(self.episode_memory["mo_objs"])

            done = False

        elif main_action == 2:
            # Acquire final image
            sted_image, bleached, conf1, conf2, fg_s, fg_c = self._acquire(imaging_action)
            mo_objs = self.mo_reward_calculator.evaluate(sted_image, conf1, conf2, fg_s, fg_c)

            reward = self.nb_reward_calculator.evaluate(sted_image, conf1, conf2, fg_s, fg_c, synapse=self.synapse)
            reward = reward * self.scale_nanodomain_reward

            done = True

        else:
            raise NotImplementedError("Only 3 actions supported...")

        # Updates memory
        self.current_step += 1
        self.episode_memory["mo_objs"].append(mo_objs)
        self.episode_memory["actions"].append(imaging_action)
        self.episode_memory["reward"].append(reward)

        state = self._update_datamap()
        self.state = state[..., numpy.newaxis]

        info = {
            "action" : action,
            "bleached" : bleached,
            "sted_image" : sted_image,
            "conf1" : conf1,
            "conf2" : conf2,
            "fg_c" : fg_c,
            "fg_s" : fg_s,
            "mo_objs" : mo_objs,
            "articulation" : self.current_articulation
        }
        if self.current_articulation == -1:
            articulation = numpy.zeros((self.spec.max_episode_steps, ))
        else:
            articulation = numpy.eye(self.spec.max_episode_steps)[self.current_articulation]

        # Build the observation space
        obs = []
        obs.extend(articulation)
        obs.extend(self.episode_memory["actions"][-1])
        obs.extend(self.episode_memory["mo_objs"][-1])
        obs = numpy.array(obs)

        return (self.state, obs), reward, done, info

    def reset(self, seed=None):
        """
        Resets the environment with a new datamap

        :returns : A `numpy.ndarray` of the molecules
        """
        super().reset(seed=seed)
        # Updates the current bleach function
        self.microscope = self.microscope_generator.generate_microscope(
            phy_react=self.bleach_sampler.sample()
        )

        self.num_request_left = self.max_num_requests
        self.current_step = 0
        self.current_articulation = -1
        self.episode_memory = {
            "actions" : [],
            "mo_objs" : [],
            "reward" : [],
        }

        state = self._update_datamap()

        self.state = state[..., numpy.newaxis]
        return (self.state, numpy.zeros((self.observation_space[1].shape[0],)))

    def render(self, info, mode='human'):
        """
        Renders the environment

        :param info: A `dict` of data
        :param mode: A `str` of the available mode
        """
        fig, axes = pyplot.subplots(1, 3, figsize=(10,3), sharey=True, sharex=True)

        axes[0].imshow(info["conf1"])
        axes[0].set_title(f"Datamap roi")

        axes[1].imshow(info["bleached"])
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
        state = numpy.random.randint(0, 2 ** 16, self.observation_space[0].shape[:-1], dtype=numpy.uint16)
        return state

    def _acquire(self, action):

        sted_image = numpy.random.randint(0, 2 ** 16, self.observation_space[0].shape[:-1], dtype=numpy.uint16)
        bleached = numpy.random.randint(0, 2 ** 16, self.observation_space[0].shape[:-1], dtype=numpy.uint16)
        conf1 = numpy.random.randint(0, 2 ** 16, self.observation_space[0].shape[:-1], dtype=numpy.uint16)
        conf2 = numpy.random.randint(0, 2 ** 16, self.observation_space[0].shape[:-1], dtype=numpy.uint16)
        fg_s = numpy.random.randint(0, 2, self.observation_space[0].shape[:-1], dtype=bool)
        fg_c = numpy.random.randint(0, 2, self.observation_space[0].shape[:-1], dtype=bool)

        return sted_image, bleached, conf1, conf2, fg_s, fg_c

    def close(self):
        return None
