
from gym.envs.registration import registry, register, make, spec

register(
    id='STED-v0',
    entry_point='gym_sted.envs:STEDEnv',
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

register(
    id='TimedSTEDdebugBleach-v0',
    entry_point='gym_sted.envs:DebugBleachSTEDTimedEnv',
    # does this correspond to the max number of steps in an episode? If so what is this value in my case?
    max_episode_steps=3,   # leave at 3 for now
)
