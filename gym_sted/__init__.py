
from gym.envs.registration import register

register(
    id='STED-v0',
    entry_point='gym_sted.envs:STEDEnv',
)
