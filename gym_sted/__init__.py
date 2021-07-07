
from gym.envs.registration import register

register(
    id='STED-v0',
    entry_point='gym_sted.envs:STEDEnv',
)

register(
    id="STEDdebugResolutionSNR-v0",
    entry_point='gym_sted.envs:DebugResolutionSNRSTEDEnv',
    max_episode_steps=3
)
