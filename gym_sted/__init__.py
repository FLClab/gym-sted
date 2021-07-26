
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
    id="STEDranking-v0",
    entry_point="gym_sted.envs:rankSTEDEnv",
    max_episode_steps=10,
    kwargs={
        "reward_calculator" : "BoundedRewardCalculator",
        "actions" : ["p_sted"]
    }
)

register(
    id="STEDranking-v1",
    entry_point="gym_sted.envs:rankSTEDEnv",
    max_episode_steps=10,
    kwargs={
        "reward_calculator" : "BoundedRewardCalculator",
        "actions" : ["p_sted", "p_ex", "pdt"]
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
    id='TimedSTEDdebugBleach-v0',
    entry_point='gym_sted.envs:DebugBleachSTEDTimedEnv',
    # does this correspond to the max number of steps in an episode? If so what is this value in my case?
    # for now the pdt is 100us for a 64x64 dmap, which means 409600 time steps per action, which means
    # the agent can complete 1 action and start another one before the episode is over
    max_episode_steps=20,   # for now the pdt is 100us for a 64x64 dmap, will not go over 13 acqs
)
