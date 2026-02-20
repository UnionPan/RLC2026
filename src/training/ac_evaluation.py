"""
Evaluation utilities for Almgren-Chriss multi-agent execution.

Provides:
- Comparison against Nash equilibrium and TWAP baselines
- Implementation shortfall analysis
- Trajectory visualization
- Statistical testing

Author: Yunian Pan
Email: yp1170@nyu.edu
"""

from typing import Dict, Any, Optional, List, Callable
import numpy as np
import torch
from dataclasses import dataclass
from pathlib import Path
import sys

# Add paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.fin.simulations import make_multi_agent_ac_env, DiscretizeActionWrapper
from lib.fin.agents import (
    create_nash_multi_agent_policy,
    create_cooperative_multi_agent_policy,
    MultiAgentPolicy,
    TWAPAgent,
    OptimalACAgent,
)


@dataclass
class EvaluationResult:
    """Results from policy evaluation."""
    mean_reward: float
    std_reward: float
    mean_shortfall_bps: float
    std_shortfall_bps: float
    episode_rewards: List[float]
    episode_shortfalls: List[float]
    inventory_trajectories: Optional[np.ndarray] = None


def evaluate_learned_policy(
    trainer,
    n_episodes: int = 100,
    deterministic: bool = True,
    collect_trajectories: bool = False,
) -> EvaluationResult:
    """
    Evaluate a learned MAPPO policy.

    Args:
        trainer: ACMAPPOTrainer instance
        n_episodes: Number of evaluation episodes
        deterministic: Use deterministic actions
        collect_trajectories: Store inventory trajectories

    Returns:
        EvaluationResult with statistics
    """
    env = trainer.env
    agent_ids = trainer.agent_ids

    episode_rewards = []
    episode_shortfalls = []
    all_trajectories = [] if collect_trajectories else None

    for ep in range(n_episodes):
        obs, infos = env.reset(seed=50000 + ep)
        trainer.reset_episode_state()

        episode_reward = 0.0
        done = False
        step = 0
        trajectory = {agent_id: [] for agent_id in agent_ids} if collect_trajectories else None

        while not done and step < trainer.config.n_steps:
            # Record inventory
            if collect_trajectories:
                for agent_id in agent_ids:
                    trajectory[agent_id].append(infos[agent_id]['inventory'])

            # Select actions
            action_results = trainer.select_actions(obs, deterministic=deterministic)
            actions = {agent_id: result['action'] for agent_id, result in action_results.items()}

            obs, rewards, terms, truncs, infos = env.step(actions)

            for agent_id in agent_ids:
                episode_reward += rewards.get(agent_id, 0.0)

                one_hot = torch.zeros(1, trainer.n_actions, device=trainer.device)
                one_hot[0, actions[agent_id]] = 1.0
                trainer.prev_actions[agent_id] = one_hot
                trainer.prev_rewards[agent_id] = torch.tensor(
                    [[rewards.get(agent_id, 0.0)]], dtype=torch.float32, device=trainer.device
                )

            done = all(terms.get(a, False) or truncs.get(a, False) for a in env.agents)
            step += 1

        episode_rewards.append(episode_reward / len(agent_ids))

        # Compute shortfall in bps
        shortfall = sum(infos[a].get('shortfall_bps', 0) for a in agent_ids) / len(agent_ids)
        episode_shortfalls.append(shortfall)

        if collect_trajectories:
            for agent_id in agent_ids:
                trajectory[agent_id].append(infos[agent_id]['inventory'])
            all_trajectories.append(trajectory)

    return EvaluationResult(
        mean_reward=np.mean(episode_rewards),
        std_reward=np.std(episode_rewards),
        mean_shortfall_bps=np.mean(episode_shortfalls),
        std_shortfall_bps=np.std(episode_shortfalls),
        episode_rewards=episode_rewards,
        episode_shortfalls=episode_shortfalls,
        inventory_trajectories=all_trajectories,
    )


def evaluate_baseline_policy(
    env,
    policy: MultiAgentPolicy,
    n_episodes: int = 100,
    collect_trajectories: bool = False,
) -> EvaluationResult:
    """
    Evaluate a baseline policy (Nash, TWAP, Cooperative).

    Args:
        env: Base multi-agent A-C environment (not discretized)
        policy: MultiAgentPolicy instance
        n_episodes: Number of episodes
        collect_trajectories: Store inventory trajectories

    Returns:
        EvaluationResult with statistics
    """
    agent_ids = env.possible_agents

    episode_rewards = []
    episode_shortfalls = []
    all_trajectories = [] if collect_trajectories else None

    for ep in range(n_episodes):
        obs, infos = env.reset(seed=60000 + ep)
        policy.reset()

        episode_reward = 0.0
        done = False
        trajectory = {agent_id: [] for agent_id in agent_ids} if collect_trajectories else None

        while not done:
            if collect_trajectories:
                for agent_id in agent_ids:
                    trajectory[agent_id].append(infos[agent_id]['inventory'])

            actions = policy.act(obs, infos)
            obs, rewards, terms, truncs, infos = env.step(actions)

            episode_reward += sum(rewards.values())
            done = all(truncs.values())

        episode_rewards.append(episode_reward / len(agent_ids))

        shortfall = sum(infos[a].get('shortfall_bps', 0) for a in agent_ids) / len(agent_ids)
        episode_shortfalls.append(shortfall)

        if collect_trajectories:
            for agent_id in agent_ids:
                trajectory[agent_id].append(infos[agent_id]['inventory'])
            all_trajectories.append(trajectory)

    return EvaluationResult(
        mean_reward=np.mean(episode_rewards),
        std_reward=np.std(episode_rewards),
        mean_shortfall_bps=np.mean(episode_shortfalls),
        std_shortfall_bps=np.std(episode_shortfalls),
        episode_rewards=episode_rewards,
        episode_shortfalls=episode_shortfalls,
        inventory_trajectories=all_trajectories,
    )


def compare_vs_baselines(
    trainer,
    n_episodes: int = 100,
    verbose: bool = True,
) -> Dict[str, EvaluationResult]:
    """
    Compare learned policy against Nash and TWAP baselines.

    Args:
        trainer: ACMAPPOTrainer instance
        n_episodes: Number of evaluation episodes
        verbose: Print results

    Returns:
        Dict mapping policy name -> EvaluationResult
    """
    results = {}

    # Evaluate learned policy
    if verbose:
        print("Evaluating learned policy...")
    results['learned'] = evaluate_learned_policy(trainer, n_episodes)

    # Evaluate Nash
    if verbose:
        print("Evaluating Nash equilibrium...")
    nash_policies = create_nash_multi_agent_policy(trainer.base_env)
    nash_policy = MultiAgentPolicy(nash_policies)
    results['nash'] = evaluate_baseline_policy(trainer.base_env, nash_policy, n_episodes)

    # Evaluate TWAP
    if verbose:
        print("Evaluating TWAP...")
    twap_policy = MultiAgentPolicy.all_twap(trainer.base_env.possible_agents)
    results['twap'] = evaluate_baseline_policy(trainer.base_env, twap_policy, n_episodes)

    # Evaluate Cooperative
    if verbose:
        print("Evaluating Cooperative optimum...")
    coop_policies = create_cooperative_multi_agent_policy(trainer.base_env)
    coop_policy = MultiAgentPolicy(coop_policies)
    results['cooperative'] = evaluate_baseline_policy(trainer.base_env, coop_policy, n_episodes)

    if verbose:
        print("\n" + "=" * 60)
        print("Evaluation Results")
        print("=" * 60)
        print(f"{'Policy':<15} {'Mean Reward':>15} {'Std':>10} {'Shortfall (bps)':>15}")
        print("-" * 60)
        for name, result in results.items():
            print(f"{name:<15} {result.mean_reward:>15.4f} {result.std_reward:>10.4f} "
                  f"{result.mean_shortfall_bps:>15.2f}")

        print("\nRelative Performance vs Baselines:")
        learned = results['learned']
        for name in ['nash', 'twap', 'cooperative']:
            diff = learned.mean_reward - results[name].mean_reward
            print(f"  vs {name}: {diff:+.4f}")

    return results


def statistical_test(
    results: Dict[str, EvaluationResult],
    baseline: str = 'nash',
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Perform statistical test comparing learned vs baseline.

    Uses Welch's t-test for unequal variances.

    Args:
        results: Dict from compare_vs_baselines
        baseline: Baseline to compare against
        alpha: Significance level

    Returns:
        Dict with test statistics
    """
    from scipy import stats

    learned = np.array(results['learned'].episode_rewards)
    baseline_rewards = np.array(results[baseline].episode_rewards)

    # Welch's t-test
    t_stat, p_value = stats.ttest_ind(learned, baseline_rewards, equal_var=False)

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((learned.std()**2 + baseline_rewards.std()**2) / 2)
    cohens_d = (learned.mean() - baseline_rewards.mean()) / pooled_std

    significant = p_value < alpha
    better = learned.mean() > baseline_rewards.mean()

    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significant': significant,
        'learned_better': significant and better,
        'learned_mean': learned.mean(),
        'baseline_mean': baseline_rewards.mean(),
        'difference': learned.mean() - baseline_rewards.mean(),
    }


def evaluate_vs_baselines(checkpoint_path: str, n_episodes: int = 100):
    """
    Load checkpoint and evaluate against baselines.

    Convenience function for command-line usage.

    Args:
        checkpoint_path: Path to checkpoint file
        n_episodes: Number of evaluation episodes
    """
    from src.training import ACMAPPOTrainer, ACMAPPOConfig

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']

    # Create trainer
    trainer = ACMAPPOTrainer(config)
    trainer.load_checkpoint(checkpoint_path)

    # Evaluate
    results = compare_vs_baselines(trainer, n_episodes, verbose=True)

    # Statistical tests
    print("\nStatistical Tests (vs Nash):")
    test_results = statistical_test(results, 'nash')
    print(f"  t-statistic: {test_results['t_statistic']:.4f}")
    print(f"  p-value: {test_results['p_value']:.4f}")
    print(f"  Cohen's d: {test_results['cohens_d']:.4f}")
    print(f"  Significant (α=0.05): {test_results['significant']}")
    print(f"  Learned better: {test_results['learned_better']}")

    return results
