
from gym.envs.registration import registry, register, make, spec

register(
    id='STED-v0',
    entry_point='gym_sted.envs:STEDEnv',
)

register(
    id='STEDdebug-v0',
    entry_point='gym_sted.envs:DebugSTEDEnv',
    max_episode_steps=3,
)
