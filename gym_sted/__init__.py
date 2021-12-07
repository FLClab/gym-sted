
from gym.envs.registration import registry, register, make, spec

############################################
# HUMAN CONTROL
############################################

register(
    id="MOSTED-human-easy-v0",
    entry_point="gym_sted.envs:HumanSTEDMultiObjectivesEnv",
    max_episode_steps=10,
    kwargs={
        "actions" : ["p_sted", "p_ex", "pdt"],
        "bleach_sampling" : "constant",
        "scale_nanodomain_reward" : 1.,
        "normalize_observations" : False
    }
)

############################################
# F1-SCORE
############################################

register(
    id="MOSTEDRankingWithExpertDemonstrationsF1Score-easy-v0",
    entry_point="gym_sted.envs:ExpertDemonstrationF1ScoreSTEDMultiObjectivesEnv",
    max_episode_steps=10,
    kwargs={
        "actions" : ["p_sted", "p_ex", "pdt"],
        "bleach_sampling" : "constant",
        "scale_nanodomain_reward" : 1.,
        "normalize_observations" : True
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
