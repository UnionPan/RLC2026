"""
Integration tests for Almgren-Chriss MAPPO training pipeline.

Tests:
1. DiscretizeActionWrapper
2. ParallelEnvTrainer
3. ACMAPPOTrainer
4. Evaluation utilities

Author: Yunian Pan
Email: yp1170@nyu.edu
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lib.fin.simulations import (
    make_multi_agent_ac_env,
    DiscretizeActionWrapper,
    discretize_action_space,
)
from lib.fin.agents import (
    create_nash_multi_agent_policy,
    MultiAgentPolicy,
)


def test_discretize_wrapper_basic():
    """Test basic discretize wrapper functionality."""
    base_env = make_multi_agent_ac_env(n_agents=2, n_steps=10)
    env = DiscretizeActionWrapper(base_env, n_bins=21)

    # Check action space changed
    from gymnasium import spaces
    assert isinstance(env.action_space('trader_0'), spaces.Discrete)
    assert env.action_space('trader_0').n == 21

    # Check observation space unchanged
    assert env.observation_space('trader_0') == base_env.observation_space('trader_0')

    env.close()


def test_discretize_wrapper_step():
    """Test stepping with discrete actions."""
    base_env = make_multi_agent_ac_env(n_agents=2, n_steps=10)
    env = DiscretizeActionWrapper(base_env, n_bins=21)

    obs, infos = env.reset(seed=42)

    # Action 10 = 50% of inventory (bin 10 out of 21 = 0.5)
    actions = {'trader_0': 10, 'trader_1': 10}
    obs, rewards, terms, truncs, infos = env.step(actions)

    # Should have rewards
    assert 'trader_0' in rewards
    assert 'trader_1' in rewards

    # Should have reduced inventory (started with 500k, sold 50%)
    assert infos['trader_0']['inventory'] < 500_000

    env.close()


def test_discretize_wrapper_fractions():
    """Test that fractions map correctly."""
    base_env = make_multi_agent_ac_env(n_agents=2, n_steps=10)
    env = DiscretizeActionWrapper(base_env, n_bins=11)  # 0%, 10%, ..., 100%

    assert len(env.fractions) == 11
    assert abs(env.fractions[0] - 0.0) < 1e-6
    assert abs(env.fractions[5] - 0.5) < 1e-6
    assert abs(env.fractions[10] - 1.0) < 1e-6

    env.close()


def test_factory_function():
    """Test discretize_action_space factory."""
    base_env = make_multi_agent_ac_env(n_agents=2, n_steps=10)
    env = discretize_action_space(base_env, n_bins=21)

    assert isinstance(env, DiscretizeActionWrapper)
    assert env.n_bins == 21

    env.close()


def test_ac_mappo_trainer_init():
    """Test ACMAPPOTrainer initialization."""
    from src.training import ACMAPPOTrainer, ACMAPPOConfig

    config = ACMAPPOConfig(
        n_agents=2,
        n_steps=10,
        n_episodes=5,
        update_freq=5,
        device='cpu',
    )

    trainer = ACMAPPOTrainer(config)

    assert trainer.n_agents == 2
    assert trainer.n_actions == 21  # Default bins
    assert len(trainer.agent_ids) == 2
    assert len(trainer.info_encoders) == 2
    assert len(trainer.policies) == 2
    assert len(trainer.critics) == 2

    trainer.env.close()


def test_ac_mappo_trainer_collect_episode():
    """Test episode collection."""
    from src.training import ACMAPPOTrainer, ACMAPPOConfig

    config = ACMAPPOConfig(
        n_agents=2,
        n_steps=10,
        n_episodes=1,
        update_freq=1,
        device='cpu',
    )

    trainer = ACMAPPOTrainer(config)
    stats = trainer.collect_episode(seed=42)

    assert 'trader_0_return' in stats
    assert 'trader_1_return' in stats
    assert 'episode_length' in stats
    assert stats['episode_length'] > 0

    # Check buffer has data
    assert len(trainer.episode_buffer) == 1

    trainer.env.close()


def test_ac_mappo_trainer_update():
    """Test policy update."""
    from src.training import ACMAPPOTrainer, ACMAPPOConfig

    config = ACMAPPOConfig(
        n_agents=2,
        n_steps=10,
        n_episodes=3,
        update_freq=3,
        n_ppo_epochs=2,
        device='cpu',
    )

    trainer = ACMAPPOTrainer(config)

    # Collect episodes
    for i in range(3):
        trainer.collect_episode(seed=i)

    # Update should work
    metrics = trainer.update()

    assert 'policy_loss' in metrics
    assert 'value_loss' in metrics
    assert 'entropy' in metrics

    # Buffer should be cleared
    assert len(trainer.episode_buffer) == 0

    trainer.env.close()


def test_ac_mappo_trainer_evaluate():
    """Test evaluation."""
    from src.training import ACMAPPOTrainer, ACMAPPOConfig

    config = ACMAPPOConfig(
        n_agents=2,
        n_steps=10,
        n_episodes=1,
        device='cpu',
    )

    trainer = ACMAPPOTrainer(config)
    stats = trainer.evaluate(n_episodes=3)

    assert 'trader_0_mean' in stats
    assert 'trader_1_mean' in stats

    trainer.env.close()


def test_ac_mappo_trainer_vs_baselines():
    """Test baseline comparison."""
    from src.training import ACMAPPOTrainer, ACMAPPOConfig

    config = ACMAPPOConfig(
        n_agents=2,
        n_steps=10,
        n_episodes=1,
        device='cpu',
    )

    trainer = ACMAPPOTrainer(config)
    results = trainer.evaluate_vs_baselines(n_episodes=3)

    assert 'learned_mean' in results
    assert 'nash_mean' in results
    assert 'twap_mean' in results
    assert 'vs_nash' in results
    assert 'vs_twap' in results

    trainer.env.close()


def test_ac_mappo_trainer_short_training():
    """Test short training run."""
    from src.training import ACMAPPOTrainer, ACMAPPOConfig

    config = ACMAPPOConfig(
        n_agents=2,
        n_steps=10,
        n_episodes=10,
        update_freq=5,
        eval_freq=100,  # Don't eval during short run
        save_freq=100,  # Don't save during short run
        device='cpu',
    )

    trainer = ACMAPPOTrainer(config)
    episode_rewards, training_metrics = trainer.train(progress_bar=False)

    # Should have collected all episodes
    assert len(episode_rewards) == 10

    # Should have done 2 updates (at episode 5 and 10)
    assert len(training_metrics) == 2

    trainer.env.close()


def test_evaluation_utilities():
    """Test evaluation utilities."""
    from src.training import ACMAPPOTrainer, ACMAPPOConfig
    from src.training import evaluate_learned_policy, compare_vs_baselines

    config = ACMAPPOConfig(
        n_agents=2,
        n_steps=10,
        n_episodes=1,
        device='cpu',
    )

    trainer = ACMAPPOTrainer(config)

    # Evaluate learned
    result = evaluate_learned_policy(trainer, n_episodes=3)
    assert result.mean_reward is not None
    assert len(result.episode_rewards) == 3

    trainer.env.close()


def test_checkpoint_save_load():
    """Test saving and loading checkpoints."""
    import tempfile
    from src.training import ACMAPPOTrainer, ACMAPPOConfig

    config = ACMAPPOConfig(
        n_agents=2,
        n_steps=10,
        n_episodes=5,
        update_freq=5,
        device='cpu',
    )

    # Create and train
    trainer1 = ACMAPPOTrainer(config)
    trainer1.train(progress_bar=False)

    # Save checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        config.save_path = tmpdir
        trainer1.save_checkpoint(5)

        # Load into new trainer
        trainer2 = ACMAPPOTrainer(config)
        trainer2.load_checkpoint(f"{tmpdir}/checkpoint_ep5.pt")

        # Evaluate both - should give same results with same seed
        result1 = trainer1.evaluate(n_episodes=3)
        result2 = trainer2.evaluate(n_episodes=3)

        # Results should be similar (not exact due to randomness in eval)
        assert abs(result1['trader_0_mean'] - result2['trader_0_mean']) < 0.1

    trainer1.env.close()
    trainer2.env.close()


if __name__ == '__main__':
    print("Running A-C Training integration tests...")

    test_discretize_wrapper_basic()
    print("✓ test_discretize_wrapper_basic")

    test_discretize_wrapper_step()
    print("✓ test_discretize_wrapper_step")

    test_discretize_wrapper_fractions()
    print("✓ test_discretize_wrapper_fractions")

    test_factory_function()
    print("✓ test_factory_function")

    test_ac_mappo_trainer_init()
    print("✓ test_ac_mappo_trainer_init")

    test_ac_mappo_trainer_collect_episode()
    print("✓ test_ac_mappo_trainer_collect_episode")

    test_ac_mappo_trainer_update()
    print("✓ test_ac_mappo_trainer_update")

    test_ac_mappo_trainer_evaluate()
    print("✓ test_ac_mappo_trainer_evaluate")

    test_ac_mappo_trainer_vs_baselines()
    print("✓ test_ac_mappo_trainer_vs_baselines")

    test_ac_mappo_trainer_short_training()
    print("✓ test_ac_mappo_trainer_short_training")

    test_evaluation_utilities()
    print("✓ test_evaluation_utilities")

    test_checkpoint_save_load()
    print("✓ test_checkpoint_save_load")

    print("\nAll tests passed!")
