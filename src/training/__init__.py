"""Training infrastructure for multi-agent RL."""

from .buffer import RolloutBuffer, MultiAgentRolloutBuffer, ReplayBuffer
from .trainer import MultiAgentTrainer
from .parallel_trainer import ParallelEnvTrainer, ParallelTrainerConfig
from .ac_mappo_trainer import ACMAPPOTrainer, ACMAPPOConfig
from .ac_evaluation import (
    EvaluationResult,
    evaluate_learned_policy,
    evaluate_baseline_policy,
    compare_vs_baselines,
    statistical_test,
    evaluate_vs_baselines,
)
from .utils import (
    get_device,
    compute_gae,
    normalize_advantages,
    RunningMeanStd,
    RewardNormalizer,
    explained_variance,
    polyak_update,
    hard_update,
    LinearSchedule,
    EpsilonScheduler,
    set_seed,
)

__all__ = [
    'RolloutBuffer',
    'MultiAgentRolloutBuffer',
    'ReplayBuffer',
    'MultiAgentTrainer',
    'ParallelEnvTrainer',
    'ParallelTrainerConfig',
    'ACMAPPOTrainer',
    'ACMAPPOConfig',
    'EvaluationResult',
    'evaluate_learned_policy',
    'evaluate_baseline_policy',
    'compare_vs_baselines',
    'statistical_test',
    'evaluate_vs_baselines',
    'get_device',
    'compute_gae',
    'normalize_advantages',
    'RunningMeanStd',
    'RewardNormalizer',
    'explained_variance',
    'polyak_update',
    'hard_update',
    'LinearSchedule',
    'EpsilonScheduler',
    'set_seed',
]
