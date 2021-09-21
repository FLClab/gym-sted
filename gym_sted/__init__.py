
from gym.envs.registration import registry, register, make, spec

# Control environment
register(
    id='STEDsum-v0',
    entry_point='gym_sted.envs:STEDEnv',
    max_episode_steps=10,
    kwargs={
        "reward_calculator" : "SumRewardCalculator"
    }
)

register(
    id='STEDmultiply-v0',
    entry_point='gym_sted.envs:STEDEnv',
    max_episode_steps=10,
    kwargs={
        "reward_calculator" : "MultiplyRewardCalculator"
    }
)

register(
    id='STEDmultiply-v1',
    entry_point='gym_sted.envs:STEDEnv',
    max_episode_steps=10,
    kwargs={
        "reward_calculator" : "MultiplyRewardCalculator",
        "actions" : ["p_sted", "p_ex", "pdt"]
    }
)

register(
    id='STEDmultiplyWithoutVision-v0',
    entry_point='gym_sted.envs:STEDEnvWithoutVision',
    max_episode_steps=10,
    kwargs={
        "reward_calculator" : "MultiplyRewardCalculator"
    }
)

register(
    id='STEDMultiplyWithDelayedReward-v0',
    entry_point='gym_sted.envs:STEDEnvWithDelayedReward',
    max_episode_steps=10,
    kwargs={
        "reward_calculator" : "MultiplyRewardCalculator",
        "actions" : ["p_sted"]
    }
)

register(
    id='STEDbounded-v0',
    entry_point='gym_sted.envs:STEDEnv',
    max_episode_steps=10,
    kwargs={
        "reward_calculator" : "BoundedRewardCalculator"
    }
)

register(
    id='STEDbounded-v1',
    entry_point='gym_sted.envs:STEDEnv',
    max_episode_steps=10,
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
    id="MOSTEDRankingWithArticulation-easy-v0",
    entry_point="gym_sted.envs:rankSTEDMultiObjectivesWithArticulationEnv",
    max_episode_steps=10,
    kwargs={
        "actions" : ["p_sted"],
        "bleach_sampling" : "constant"
    }
)

register(
    id="MOSTEDRankingWithArticulation-easy-v1",
    entry_point="gym_sted.envs:rankSTEDMultiObjectivesWithArticulationEnv",
    max_episode_steps=10,
    kwargs={
        "actions" : ["p_sted", "p_ex", "pdt"],
        "bleach_sampling" : "constant"
    }
)

register(
    id="MOSTEDRankingWithArticulation-easy-v2",
    entry_point="gym_sted.envs:rankSTEDMultiObjectivesWithArticulationEnv",
    max_episode_steps=10,
    kwargs={
        "actions" : ["p_sted", "p_ex", "pdt"],
        "bleach_sampling" : "constant",
        "select_final" : False
    }
)

register(
    id="MOSTEDRankingWithArticulation-easy-v3",
    entry_point="gym_sted.envs:rankSTEDMultiObjectivesWithArticulationEnv",
    max_episode_steps=10,
    kwargs={
        "actions" : ["p_sted", "p_ex", "pdt"],
        "bleach_sampling" : "constant",
        "select_final" : False,
        "scale_rank_reward" : True
    }
)

register(
    id="MOSTEDRankingWithArticulation-easy-v4",
    entry_point="gym_sted.envs:rankSTEDMultiObjectivesWithArticulationEnv",
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
    id="MOSTEDRankingWithArticulation-hard-v0",
    entry_point="gym_sted.envs:rankSTEDMultiObjectivesWithArticulationEnv",
    max_episode_steps=10,
    kwargs={
        "actions" : ["p_sted"],
        "bleach_sampling" : "normal"
    }
)

register(
    id="MOSTEDRankingWithArticulation-hard-v1",
    entry_point="gym_sted.envs:rankSTEDMultiObjectivesWithArticulationEnv",
    max_episode_steps=10,
    kwargs={
        "actions" : ["p_sted", "p_ex", "pdt"],
        "bleach_sampling" : "normal",
        "scale_rank_reward" : False
    }
)

register(
    id="MOSTEDRankingWithArticulation-hard-v2",
    entry_point="gym_sted.envs:rankSTEDMultiObjectivesWithArticulationEnv",
    max_episode_steps=10,
    kwargs={
        "actions" : ["p_sted", "p_ex", "pdt"],
        "bleach_sampling" : "normal",
        "select_final" : False,
        "scale_rank_reward" : False
    }
)

register(
    id="MOSTEDRankingWithArticulation-hard-v3",
    entry_point="gym_sted.envs:rankSTEDMultiObjectivesWithArticulationEnv",
    max_episode_steps=10,
    kwargs={
        "actions" : ["p_sted", "p_ex", "pdt"],
        "bleach_sampling" : "normal",
        "select_final" : False,
        "scale_rank_reward" : True
    }
)

register(
    id="MOSTEDRankingWithArticulation-hard-v4",
    entry_point="gym_sted.envs:rankSTEDMultiObjectivesWithArticulationEnv",
    max_episode_steps=10,
    kwargs={
        "actions" : ["p_sted", "p_ex", "pdt"],
        "bleach_sampling" : "normal",
        "select_final" : False,
        "scale_rank_reward" : False,
        "scale_nanodomain_reward" : 1.,
    }
)

register(
    id="MOSTEDRanking-easy-v0",
    entry_point="gym_sted.envs:rankSTEDMultiObjectivesEnv",
    max_episode_steps=10,
    kwargs={
        "actions" : ["p_sted", "p_ex", "pdt"],
        "bleach_sampling" : "constant",
        "scale_nanodomain_reward" : 1.,
    }
)

register(
    id="MOSTEDRanking-mid-v0",
    entry_point="gym_sted.envs:rankSTEDMultiObjectivesEnv",
    max_episode_steps=10,
    kwargs={
        "actions" : ["p_sted", "p_ex", "pdt"],
        "bleach_sampling" : "choice",
        "scale_nanodomain_reward" : 1.,
    }
)

register(
    id="MOSTEDRanking-hard-v0",
    entry_point="gym_sted.envs:rankSTEDMultiObjectivesEnv",
    max_episode_steps=10,
    kwargs={
        "actions" : ["p_sted", "p_ex", "pdt"],
        "bleach_sampling" : "normal",
        "scale_nanodomain_reward" : 1.,
    }
)

register(
    id="MOSTEDRankingWithDelayedReward-easy-v0",
    entry_point="gym_sted.envs:rankSTEDMultiObjectivesWithDelayedRewardEnv",
    max_episode_steps=10,
    kwargs={
        "actions" : ["p_sted", "p_ex", "pdt"],
        "bleach_sampling" : "constant",
        "scale_nanodomain_reward" : 1.,
    }
)

register(
    id="MOSTEDRankingWithDelayedReward-mid-v0",
    entry_point="gym_sted.envs:rankSTEDMultiObjectivesWithDelayedRewardEnv",
    max_episode_steps=10,
    kwargs={
        "actions" : ["p_sted", "p_ex", "pdt"],
        "bleach_sampling" : "choice",
        "scale_nanodomain_reward" : 1.,
    }
)

register(
    id="MOSTEDRankingWithDelayedReward-hard-v0",
    entry_point="gym_sted.envs:rankSTEDMultiObjectivesWithDelayedRewardEnv",
    max_episode_steps=10,
    kwargs={
        "actions" : ["p_sted", "p_ex", "pdt"],
        "bleach_sampling" : "normal",
        "scale_nanodomain_reward" : 1.,
    }
)

register(
    id="MOSTEDRanking-easy-v1",
    entry_point="gym_sted.envs:rankSTEDMultiObjectivesEnv",
    max_episode_steps=10,
    kwargs={
        "actions" : ["p_sted", "p_ex", "pdt"],
        "bleach_sampling" : "constant",
        "scale_nanodomain_reward" : 3.,
    }
)

register(
    id="MOSTEDRanking-mid-v1",
    entry_point="gym_sted.envs:rankSTEDMultiObjectivesEnv",
    max_episode_steps=10,
    kwargs={
        "actions" : ["p_sted", "p_ex", "pdt"],
        "bleach_sampling" : "choice",
        "scale_nanodomain_reward" : 3.,
    }
)

register(
    id="MOSTEDRanking-hard-v1",
    entry_point="gym_sted.envs:rankSTEDMultiObjectivesEnv",
    max_episode_steps=10,
    kwargs={
        "actions" : ["p_sted", "p_ex", "pdt"],
        "bleach_sampling" : "normal",
        "scale_nanodomain_reward" : 3.,
    }
)

############################################
# CONTEXTUAL
############################################

register(
    id="ContextualMOSTED-easy-v0",
    entry_point="gym_sted.envs:ContextualSTEDMultiObjectivesEnv",
    max_episode_steps=10,
    kwargs={
        "actions" : ["p_sted", "p_ex", "pdt"],
        "bleach_sampling" : "constant",
        "scale_nanodomain_reward" : 1.,
    }
)

register(
    id="ContextualMOSTED-mid-v0",
    entry_point="gym_sted.envs:ContextualSTEDMultiObjectivesEnv",
    max_episode_steps=10,
    kwargs={
        "actions" : ["p_sted", "p_ex", "pdt"],
        "bleach_sampling" : "choice",
        "scale_nanodomain_reward" : 1.,
    }
)

register(
    id="ContextualMOSTED-hard-v0",
    entry_point="gym_sted.envs:ContextualSTEDMultiObjectivesEnv",
    max_episode_steps=10,
    kwargs={
        "actions" : ["p_sted", "p_ex", "pdt"],
        "bleach_sampling" : "normal",
        "scale_nanodomain_reward" : 1.,
    }
)

register(
    id="ContextualRankingMOSTED-easy-v0",
    entry_point="gym_sted.envs:ContextualRankingSTEDMultiObjectivesEnv",
    max_episode_steps=10,
    kwargs={
        "actions" : ["p_sted", "p_ex", "pdt"],
        "bleach_sampling" : "constant",
    }
)

register(
    id="ContextualRankingMOSTED-mid-v0",
    entry_point="gym_sted.envs:ContextualRankingSTEDMultiObjectivesEnv",
    max_episode_steps=10,
    kwargs={
        "actions" : ["p_sted", "p_ex", "pdt"],
        "bleach_sampling" : "choice"
    }
)

register(
    id="ContextualRankingMOSTED-hard-v0",
    entry_point="gym_sted.envs:ContextualRankingSTEDMultiObjectivesEnv",
    max_episode_steps=10,
    kwargs={
        "actions" : ["p_sted", "p_ex", "pdt"],
        "bleach_sampling" : "normal"
    }
)

# Ranking environment recurrent
register(
    id="MOSTEDrankingWithArticulation-recurrent-easy-v0",
    entry_point="gym_sted.envs:rankSTEDRecurrentMultiObjectivesWithArticulationEnv",
    max_episode_steps=10,
    kwargs={
        "actions" : ["p_sted", "p_ex", "pdt"],
        "bleach_sampling" : "constant",
        "select_final" : False,
        "scale_rank_reward" : True
    }
)

register(
    id="MOSTEDrankingWithArticulation-recurrent-hard-v0",
    entry_point="gym_sted.envs:rankSTEDRecurrentMultiObjectivesWithArticulationEnv",
    max_episode_steps=10,
    kwargs={
        "actions" : ["p_sted", "p_ex", "pdt"],
        "bleach_sampling" : "normal",
        "select_final" : False,
        "scale_rank_reward" : True
    }
)

register(
    id="MOSTEDrankingWithArticulation-recurrent-easy-v1",
    entry_point="gym_sted.envs:rankSTEDRecurrentMultiObjectivesWithArticulationEnv",
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
    id="MOSTEDrankingWithArticulation-recurrent-hard-v1",
    entry_point="gym_sted.envs:rankSTEDRecurrentMultiObjectivesWithArticulationEnv",
    max_episode_steps=10,
    kwargs={
        "actions" : ["p_sted", "p_ex", "pdt"],
        "bleach_sampling" : "normal",
        "select_final" : False,
        "scale_rank_reward" : False,
        "scale_nanodomain_reward" : 1.
    }
)

register(
    id="MOSTEDRanking-recurrent-easy-v0",
    entry_point="gym_sted.envs:rankSTEDRecurrentMultiObjectivesEnv",
    max_episode_steps=10,
    kwargs={
        "actions" : ["p_sted"],
        "bleach_sampling" : "constant",
        "scale_nanodomain_reward" : 1.,
    }
)

register(
    id="MOSTEDRanking-recurrent-easy-v1",
    entry_point="gym_sted.envs:rankSTEDRecurrentMultiObjectivesEnv",
    max_episode_steps=10,
    kwargs={
        "actions" : ["p_sted", "p_ex", "pdt"],
        "bleach_sampling" : "constant",
        "scale_nanodomain_reward" : 1.,
    }
)

register(
    id="MOSTEDRanking-recurrent-mid-v0",
    entry_point="gym_sted.envs:rankSTEDRecurrentMultiObjectivesEnv",
    max_episode_steps=10,
    kwargs={
        "actions" : ["p_sted", "p_ex", "pdt"],
        "bleach_sampling" : "choice",
        "scale_nanodomain_reward" : 1.,
    }
)

register(
    id="MOSTEDRanking-recurrent-hard-v0",
    entry_point="gym_sted.envs:rankSTEDRecurrentMultiObjectivesEnv",
    max_episode_steps=10,
    kwargs={
        "actions" : ["p_sted", "p_ex", "pdt"],
        "bleach_sampling" : "normal",
        "scale_nanodomain_reward" : 1.,
    }
)

# Debug env for Timed exp, everything is the same except that bleach is OFF (15/09/21)
register(
    id="STEDtimed-exp-debug-v0",
    entry_point="gym_sted.envs:timedExpSTEDEnvDebug",
    max_episode_steps=50,   # for exp_time_us=2000000 the max number of steps is 48, set to 50 in case
    kwargs={
        "reward_calculator": "NanodomainsRewardCalculator",
        "actions": ["pdt", "p_ex", "p_sted"],
        "bleach_sampling": "constant",
        "flash_mode": "exp"
    }
)

# Timed env for exp 2
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
    id='STEDdebugRecurrent-v0',
    entry_point='gym_sted.envs:DebugRankSTEDRecurrentMultiObjectivesEnv',
    max_episode_steps=10,
    kwargs={
        "actions" : ["p_sted", "p_ex", "pdt"],
        "bleach_sampling" : "normal",
        "select_final" : False,
        "scale_rank_reward" : False,
        "scale_nanodomain_reward" : 1.
    }
)


register(
    id='TimedSTEDdebugBleach-v0',
    entry_point='gym_sted.envs:DebugBleachSTEDTimedEnv',
    # does this correspond to the max number of steps in an episode? If so what is this value in my case?
    # for now the pdt is 100us for a 64x64 dmap, which means 409600 time steps per action, which means
    # the agent can complete 1 action and start another one before the episode is over
    max_episode_steps=20,   # for now the pdt is 100us for a 64x64 dmap, will not go over 13 acqs
)
