
from gym.envs.registration import registry, register, make, spec

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
