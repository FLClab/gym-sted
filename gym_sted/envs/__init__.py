
import logging

from gym_sted.envs.sted_env import STEDEnv, STEDEnvWithoutVision, STEDEnvWithDelayedReward
from gym_sted.envs.ranking_sted_env import (
    rankSTEDSingleObjectiveEnv,
    rankSTEDMultiObjectivesEnv,
    rankSTEDMultiObjectivesWithArticulationEnv,
    rankSTEDMultiObjectivesWithDelayedRewardEnv,
)
from gym_sted.envs.contextual_sted_env import (
    ContextualSTEDMultiObjectivesEnv,
    ContextualRankingSTEDMultiObjectivesEnv,
)
from gym_sted.envs.expert_sted_env import (
    ExpertDemonstrationSTEDMultiObjectivesEnv,
    ExpertDemonstrationF1ScoreSTEDMultiObjectivesEnv
)
from gym_sted.envs.recurrent_sted_env import (
    rankSTEDRecurrentMultiObjectivesEnv,
    rankSTEDRecurrentMultiObjectivesWithArticulationEnv,
    ContextualRecurrentSTEDMultiObjectivesEnv,
)
from gym_sted.envs.sequence_sted_env import (
    SequenceSTEDMultiObjectivesEnv
)
from gym_sted.envs.preference_sted_env import (
    PreferenceSTEDMultiObjectivesEnv,
    PreferenceCountRateScaleSTEDMultiObjectivesEnv,
    PreferenceCountRateScaleRewardEngSTEDMultiObjectivesEnv,
    RecurrentPreferenceCountRateScaleSTEDMultiObjectivesEnv
)
try:
    from gym_sted.envs.abberior_env import (
        AbberiorSTEDMultiObjectivesEnv,
        AbberiorSTEDCountRateMultiObjectivesEnv
    )
except ModuleNotFoundError as e:
    logging.warn(e)
    logging.warn("Not all required packages are installed for Abberior envs...")
    logging.warn("Using it will raise errors!")    
    
from gym_sted.envs.timed_sted_env import timedExpSTEDEnv, timedExpSTEDEnvBleach
from gym_sted.envs.timed_sted_env_debug import timedExpSTEDEnvDebug
from gym_sted.envs.pre_traj_debugging_env import preTrajDebugEnv, preTrajDebugEnvLin, preTrajDebugEnvExp
from gym_sted.envs.debug_env import DebugResolutionSNRSTEDEnv, DebugBleachSTEDEnv, DebugBleachSTEDTimedEnv
