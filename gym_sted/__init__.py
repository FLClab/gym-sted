
from gym.envs.registration import registry, register, make, spec

# Control environment
register(
    id='STEDsum-v0',
    entry_point='gym_sted.envs:STEDEnv',
    kwargs={
        "reward_calculator" : "SumRewardCalculator"
    }
)

register(
    id='STEDmultiply-v0',
    entry_point='gym_sted.envs:STEDEnv',
    kwargs={
        "reward_calculator" : "MultiplyRewardCalculator"
    }
)

register(
    id='STEDbounded-v0',
    entry_point='gym_sted.envs:STEDEnv',
    kwargs={
        "reward_calculator" : "BoundedRewardCalculator"
    }
)

register(
    id='STEDbounded-v1',
    entry_point='gym_sted.envs:STEDEnv',
    kwargs={
        "reward_calculator" : "BoundedRewardCalculator",
        "actions" : ["p_sted", "p_ex", "pdt"]
    }
)

# Ranking environment
register(
    id="STEDranking-easy-v0",
    entry_point="gym_sted.envs:rankSTEDSingleObjectiveEnv",
    max_episode_steps=10,
    kwargs={
        "reward_calculator" : "MultiplyRewardCalculator",
        "actions" : ["p_sted"]
    }
)

register(
    id="STEDranking-easy-v1",
    entry_point="gym_sted.envs:rankSTEDSingleObjectiveEnv",
    max_episode_steps=10,
    kwargs={
        "reward_calculator" : "MultiplyRewardCalculator",
        "actions" : ["p_sted", "p_ex", "pdt"]
    }
)

register(
    id="STEDranking-hard-v0",
    entry_point="gym_sted.envs:rankSTEDSingleObjectiveEnv",
    max_episode_steps=10,
    kwargs={
        "reward_calculator" : "BoundedRewardCalculator",
        "actions" : ["p_sted"]
    }
)

register(
    id="STEDranking-hard-v1",
    entry_point="gym_sted.envs:rankSTEDSingleObjectiveEnv",
    max_episode_steps=10,
    kwargs={
        "reward_calculator" : "BoundedRewardCalculator",
        "actions" : ["p_sted", "p_ex", "pdt"]
    }
)

register(
    id="MOSTEDranking-easy-v0",
    entry_point="gym_sted.envs:rankSTEDMultiObjectivesEnv",
    max_episode_steps=10,
    kwargs={
        "actions" : ["p_sted"],
        "bleach_sampling" : "constant"
    }
)

register(
    id="MOSTEDranking-easy-v1",
    entry_point="gym_sted.envs:rankSTEDMultiObjectivesEnv",
    max_episode_steps=10,
    kwargs={
        "actions" : ["p_sted", "p_ex", "pdt"],
        "bleach_sampling" : "constant"
    }
)

register(
    id="MOSTEDranking-easy-v2",
    entry_point="gym_sted.envs:rankSTEDMultiObjectivesEnv",
    max_episode_steps=10,
    kwargs={
        "actions" : ["p_sted", "p_ex", "pdt"],
        "bleach_sampling" : "constant",
        "select_final" : False
    }
)

register(
    id="MOSTEDranking-easy-v3",
    entry_point="gym_sted.envs:rankSTEDMultiObjectivesEnv",
    max_episode_steps=10,
    kwargs={
        "actions" : ["p_sted", "p_ex", "pdt"],
        "bleach_sampling" : "constant",
        "select_final" : False,
        "scale_rank_reward" : True
    }
)

register(
    id="MOSTEDranking-easy-v4",
    entry_point="gym_sted.envs:rankSTEDMultiObjectivesEnv",
    max_episode_steps=10,
    kwargs={
        "actions" : ["p_sted", "p_ex", "pdt"],
        "bleach_sampling" : "constant",
        "select_final" : False,
        "scale_rank_reward" : False,
        "scale_nanodomain_reward" : 1.
    }
)

register(
    id="MOSTEDranking-hard-v0",
    entry_point="gym_sted.envs:rankSTEDMultiObjectivesEnv",
    max_episode_steps=10,
    kwargs={
        "actions" : ["p_sted"],
        "bleach_sampling" : "normal"
    }
)

register(
    id="MOSTEDranking-hard-v1",
    entry_point="gym_sted.envs:rankSTEDMultiObjectivesEnv",
    max_episode_steps=10,
    kwargs={
        "actions" : ["p_sted", "p_ex", "pdt"],
        "bleach_sampling" : "normal",
        "scale_rank_reward" : False
    }
)

register(
    id="MOSTEDranking-hard-v2",
    entry_point="gym_sted.envs:rankSTEDMultiObjectivesEnv",
    max_episode_steps=10,
    kwargs={
        "actions" : ["p_sted", "p_ex", "pdt"],
        "bleach_sampling" : "normal",
        "select_final" : False,
        "scale_rank_reward" : False
    }
)

register(
    id="MOSTEDranking-hard-v3",
    entry_point="gym_sted.envs:rankSTEDMultiObjectivesEnv",
    max_episode_steps=10,
    kwargs={
        "actions" : ["p_sted", "p_ex", "pdt"],
        "bleach_sampling" : "normal",
        "select_final" : False,
        "scale_rank_reward" : True
    }
)

register(
    id="MOSTEDranking-hard-v4",
    entry_point="gym_sted.envs:rankSTEDMultiObjectivesEnv",
    max_episode_steps=10,
    kwargs={
        "actions" : ["p_sted", "p_ex", "pdt"],
        "bleach_sampling" : "normal",
        "select_final" : False,
        "scale_rank_reward" : False,
        "scale_nanodomain_reward" : 1.,
    }
)

# Ranking environment recurrent
register(
    id="MOSTEDranking-recurrent-easy-v0",
    entry_point="gym_sted.envs:rankSTEDRecurrentMultiObjectivesEnv",
    max_episode_steps=10,
    kwargs={
        "actions" : ["p_sted", "p_ex", "pdt"],
        "bleach_sampling" : "constant",
        "select_final" : False,
        "scale_rank_reward" : True
    }
)

register(
    id="MOSTEDranking-recurrent-hard-v0",
    entry_point="gym_sted.envs:rankSTEDRecurrentMultiObjectivesEnv",
    max_episode_steps=10,
    kwargs={
        "actions" : ["p_sted", "p_ex", "pdt"],
        "bleach_sampling" : "normal",
        "select_final" : False,
        "scale_rank_reward" : True
    }
)

register(
    id="MOSTEDranking-recurrent-easy-v1",
    entry_point="gym_sted.envs:rankSTEDRecurrentMultiObjectivesEnv",
    max_episode_steps=10,
    kwargs={
        "actions" : ["p_sted", "p_ex", "pdt"],
        "bleach_sampling" : "constant",
        "select_final" : False,
        "scale_rank_reward" : False,
        "scale_nanodomain_reward" : 1.
    }
)

register(
    id="MOSTEDranking-recurrent-hard-v1",
    entry_point="gym_sted.envs:rankSTEDRecurrentMultiObjectivesEnv",
    max_episode_steps=10,
    kwargs={
        "actions" : ["p_sted", "p_ex", "pdt"],
        "bleach_sampling" : "normal",
        "select_final" : False,
        "scale_rank_reward" : False,
        "scale_nanodomain_reward" : 1.
    }
)

# Timed envs for exp 2
register(
    id="STEDtimed-exp-easy-v5",
    entry_point="gym_sted.envs:timedExpSTEDEnv",
    max_episode_steps=50,   # for exp_time_us=2000000 the max number of steps is 48, set to 50 in case
    kwargs={
        "reward_calculator": "NanodomainsRewardCalculator",
        "actions": ["pdt", "p_ex", "p_sted"],
        "bleach_sampling": "constant",
        "flash_mode": "exp"
    }
)

register(
    id="STEDtimed-exp-easy-noisy-v5",
    entry_point="gym_sted.envs:timedExpSTEDEnv",
    max_episode_steps=50,   # for exp_time_us=2000000 the max number of steps is 48, set to 50 in case
    kwargs={
        "reward_calculator": "NanodomainsRewardCalculator",
        "actions": ["pdt", "p_ex", "p_sted"],
        "bleach_sampling": "constant",
        "flash_mode": "exp",
        "detector_noise": 100
    }
)

register(
    id="STEDtimed-exp-hard-v5",
    entry_point="gym_sted.envs:timedExpSTEDEnv",
    max_episode_steps=50,   # for exp_time_us=2000000 the max number of steps is 48, set to 50 in case
    kwargs={
        "reward_calculator": "NanodomainsRewardCalculator",
        "actions": ["pdt", "p_ex", "p_sted"],
        "bleach_sampling": "normal",
        "flash_mode": "exp"
    }
)

register(
    id="STEDtimed-exp-hard-noisy-v5",
    entry_point="gym_sted.envs:timedExpSTEDEnv",
    max_episode_steps=50,   # for exp_time_us=2000000 the max number of steps is 48, set to 50 in case
    kwargs={
        "reward_calculator": "NanodomainsRewardCalculator",
        "actions": ["pdt", "p_ex", "p_sted"],
        "bleach_sampling": "normal",
        "flash_mode": "exp",
        "detector_noise": 100
    }
)

register(
    id="STEDtimed-sampled-easy-v5",
    entry_point="gym_sted.envs:timedExpSTEDEnv",
    max_episode_steps=50,   # for exp_time_us=2000000 the max number of steps is 48, set to 50 in case
    kwargs={
        "reward_calculator": "NanodomainsRewardCalculator",
        "actions": ["pdt", "p_ex", "p_sted"],
        "bleach_sampling": "constant",
        "flash_mode": "sampled"
    }
)

register(
    id="STEDtimed-sampled-easy-noisy-v5",
    entry_point="gym_sted.envs:timedExpSTEDEnv",
    max_episode_steps=50,   # for exp_time_us=2000000 the max number of steps is 48, set to 50 in case
    kwargs={
        "reward_calculator": "NanodomainsRewardCalculator",
        "actions": ["pdt", "p_ex", "p_sted"],
        "bleach_sampling": "constant",
        "flash_mode": "sampled",
        "detector_noise": 100
    }
)

register(
    id="STEDtimed-sampled-hard-v5",
    entry_point="gym_sted.envs:timedExpSTEDEnv",
    max_episode_steps=50,   # for exp_time_us=2000000 the max number of steps is 48, set to 50 in case
    kwargs={
        "reward_calculator": "NanodomainsRewardCalculator",
        "actions": ["pdt", "p_ex", "p_sted"],
        "bleach_sampling": "normal",
        "flash_mode": "sampled"
    }
)

register(
    id="STEDtimed-sampled-hard-noisy-v5",
    entry_point="gym_sted.envs:timedExpSTEDEnv",
    max_episode_steps=50,   # for exp_time_us=2000000 the max number of steps is 48, set to 50 in case
    kwargs={
        "reward_calculator": "NanodomainsRewardCalculator",
        "actions": ["pdt", "p_ex", "p_sted"],
        "bleach_sampling": "normal",
        "flash_mode": "sampled",
        "detector_noise": 100
    }
)

"""
Follows the same implementation as v2, except the synapses are generated with random seed,
the flash is now at idx 2 - 7, randomly sampled, and there is a slight increase in signal in the nanodomains 1 step
before they flash
"""
# register(
#     id="STEDtimed-v3",
#     entry_point="gym_sted.envs:timedExpSTEDEnv2Bump",
#     max_episode_steps=50,   # for exp_time_us=500000 the max number of steps is 13, but I will prob change the exp time
#     kwargs={
#         "reward_calculator" : "NanodomainsRewardCalculator",
#         "actions" : ["pdt", "p_ex", "p_sted"],
#         "bleach_sampling": "constant"
#     }
# )
#
# register(
#     id="STEDtimed-noisy-v3",
#     entry_point="gym_sted.envs:timedExpSTEDEnv2Bump",
#     max_episode_steps=50,   # for exp_time_us=500000 the max number of steps is 13, but I will prob change the exp time
#     kwargs={
#         "reward_calculator" : "NanodomainsRewardCalculator",
#         "actions" : ["pdt", "p_ex", "p_sted"],
#         "bleach_sampling": "constant",
#         "detector_noise": 100
#     }
# )

# register(
#     id="STEDtimed-hard-v3",
#     entry_point="gym_sted.envs:timedExpSTEDEnv2Bump",
#     max_episode_steps=50,   # for exp_time_us=500000 the max number of steps is 13, but I will prob change the exp time
#     kwargs={
#         "reward_calculator" : "NanodomainsRewardCalculator",
#         "actions" : ["pdt", "p_ex", "p_sted"],
#         "bleach_sampling": "normal"
#     }
# )
#
# register(
#     id="STEDtimed-noisy-hard-v3",
#     entry_point="gym_sted.envs:timedExpSTEDEnv2Bump",
#     max_episode_steps=50,   # for exp_time_us=500000 the max number of steps is 13, but I will prob change the exp time
#     kwargs={
#         "reward_calculator" : "NanodomainsRewardCalculator",
#         "actions" : ["pdt", "p_ex", "p_sted"],
#         "bleach_sampling": "normal",
#         "detector_noise": 100
#     }
# )

"""
Follows the same implementation as v3, but the flashes are sampled from Theresa's data instead of hand crafted
"""
# register(
#     id="STEDtimed-v4",
#     entry_point="gym_sted.envs:timedExpSTEDEnv2SampledFlash",
#     max_episode_steps=50,   # for exp_time_us=500000 the max number of steps is 13, but I will prob change the exp time
#     kwargs={
#         "reward_calculator" : "NanodomainsRewardCalculator",
#         "actions" : ["pdt", "p_ex", "p_sted"],
#         "bleach_sampling": "constant"
#     }
# )

# register(
#     id="STEDtimed-noisy-v4",
#     entry_point="gym_sted.envs:timedExpSTEDEnv2SampledFlash",
#     max_episode_steps=50,   # for exp_time_us=500000 the max number of steps is 13, but I will prob change the exp time
#     kwargs={
#         "reward_calculator" : "NanodomainsRewardCalculator",
#         "actions" : ["pdt", "p_ex", "p_sted"],
#         "bleach_sampling": "constant",
#         "detector_noise": 100
#     }
# )

# register(
#     id="STEDtimed-hard-v4",
#     entry_point="gym_sted.envs:timedExpSTEDEnv2SampledFlash",
#     max_episode_steps=50,   # for exp_time_us=500000 the max number of steps is 13, but I will prob change the exp time
#     kwargs={
#         "reward_calculator" : "NanodomainsRewardCalculator",
#         "actions" : ["pdt", "p_ex", "p_sted"],
#         "bleach_sampling": "normal"
#     }
# )

# register(
#     id="STEDtimed-noisy-hard-v4",
#     entry_point="gym_sted.envs:timedExpSTEDEnv2SampledFlash",
#     max_episode_steps=50,   # for exp_time_us=500000 the max number of steps is 13, but I will prob change the exp time
#     kwargs={
#         "reward_calculator" : "NanodomainsRewardCalculator",
#         "actions" : ["pdt", "p_ex", "p_sted"],
#         "bleach_sampling": "normal",
#         "detector_noise": 100
#     }
# )

# Debug environment
register(
    id="STEDdebugResolutionSNR-v0",
    entry_point='gym_sted.envs:DebugResolutionSNRSTEDEnv',
    max_episode_steps=5
)

register(
    id='STEDdebugBleach-v0',
    entry_point='gym_sted.envs:DebugBleachSTEDEnv',
    max_episode_steps=3,
)

register(
    id='TimedSTEDdebugBleach-v0',
    entry_point='gym_sted.envs:DebugBleachSTEDTimedEnv',
    # does this correspond to the max number of steps in an episode? If so what is this value in my case?
    # for now the pdt is 100us for a 64x64 dmap, which means 409600 time steps per action, which means
    # the agent can complete 1 action and start another one before the episode is over
    max_episode_steps=20,   # for now the pdt is 100us for a 64x64 dmap, will not go over 13 acqs
)
