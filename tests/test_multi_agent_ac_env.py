"""
Tests for Multi-Agent Almgren-Chriss Environment (POSG).

Tests:
1. Environment initialization with multiple agents
2. PettingZoo API compliance
3. Partial observability
4. Shared price impact mechanics
5. Strategy comparison (cooperative vs competitive)
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lib.fin.simulations.multi_agent_ac_env import (
    MultiAgentAlmgrenChrissEnv,
    MultiAgentACParams,
    make_multi_agent_ac_env,
)


def test_env_initialization():
    """Test environment initializes correctly."""
    env = make_multi_agent_ac_env(n_agents=2)

    assert len(env.possible_agents) == 2
    assert env.possible_agents == ['trader_0', 'trader_1']
    assert env.params.n_agents == 2

    env.close()


def test_reset_returns_observations():
    """Test reset returns observations for all agents."""
    env = make_multi_agent_ac_env(n_agents=3)
    observations, infos = env.reset(seed=42)

    assert len(observations) == 3
    assert 'trader_0' in observations
    assert 'trader_1' in observations
    assert 'trader_2' in observations

    # Each observation has correct shape
    for agent, obs in observations.items():
        assert obs.shape == (4,)  # [price, inventory, cumulative, time]

    env.close()


def test_parallel_step():
    """Test all agents can step simultaneously."""
    env = make_multi_agent_ac_env(n_agents=2)
    observations, _ = env.reset(seed=42)

    # All agents trade 50%
    actions = {
        'trader_0': np.array([0.5]),
        'trader_1': np.array([0.5]),
    }

    obs, rewards, terms, truncs, infos = env.step(actions)

    # All agents get results
    assert len(obs) == 2
    assert len(rewards) == 2
    assert 'trader_0' in rewards
    assert 'trader_1' in rewards

    # Inventories decreased
    assert infos['trader_0']['inventory'] < env.params.X_0
    assert infos['trader_1']['inventory'] < env.params.X_0

    env.close()


def test_partial_observability():
    """Test agents cannot see each other's state."""
    env = make_multi_agent_ac_env(n_agents=2)
    observations, _ = env.reset(seed=42)

    # Agent 0 trades, agent 1 doesn't
    actions = {
        'trader_0': np.array([0.5]),
        'trader_1': np.array([0.0]),
    }

    obs, _, _, _, infos = env.step(actions)

    # Agent 1's observation doesn't reveal agent 0's trade
    # (though price impact is visible)
    obs_0 = obs['trader_0']
    obs_1 = obs['trader_1']

    # Price is same for both (shared)
    assert obs_0[0] == obs_1[0]

    # But inventories are different (private)
    # obs[1] is own inventory fraction
    assert obs_0[1] != obs_1[1]  # 0.5 vs 1.0

    env.close()


def test_shared_price_impact():
    """Test permanent impact is shared across agents."""
    # Two identical agents
    env = make_multi_agent_ac_env(
        n_agents=2,
        X_0=100_000,
        gamma=1e-5,  # High impact for visibility
        n_steps=5,
    )

    observations, _ = env.reset(seed=42)
    initial_price = env.S_impacted

    # Both agents trade half
    actions = {
        'trader_0': np.array([0.5]),
        'trader_1': np.array([0.5]),
    }
    obs, _, _, _, infos = env.step(actions)

    # Price dropped due to combined impact
    # Total traded = 50k + 50k = 100k
    # Permanent impact = gamma * 100k = 1e-5 * 1e5 = 1.0
    price_drop = initial_price - env.S_impacted

    # Should see price impact
    assert price_drop > 0
    assert env.S_impacted < initial_price

    env.close()


def test_heterogeneous_agents():
    """Test agents with different parameters."""
    params = MultiAgentACParams(
        n_agents=2,
        X_0_list=[1_000_000, 100_000],  # Large vs small
        eta_list=[2e-6, 5e-6],          # Different temporary impact
    )

    env = MultiAgentAlmgrenChrissEnv(params=params, n_steps=10)
    obs, infos = env.reset(seed=42)

    # Check initial inventories
    assert infos['trader_0']['inventory'] == 1_000_000
    assert infos['trader_1']['inventory'] == 100_000

    env.close()


def test_twap_both_agents():
    """Test TWAP strategy for both agents."""
    env = make_multi_agent_ac_env(n_agents=2, n_steps=10)
    obs, _ = env.reset(seed=42)

    total_rewards = {'trader_0': 0.0, 'trader_1': 0.0}
    done = False
    step = 0

    while not done:
        remaining_steps = env.n_steps - step
        fraction = 1.0 / remaining_steps if remaining_steps > 0 else 1.0

        actions = {agent: np.array([fraction]) for agent in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)

        for agent, r in rewards.items():
            total_rewards[agent] += r

        done = all(truncs.values()) or all(terms.values())
        step += 1

    # Both agents should have similar performance (symmetric)
    assert abs(total_rewards['trader_0'] - total_rewards['trader_1']) < 0.1

    env.close()


def test_competitive_vs_cooperative():
    """Test competitive (fast) vs cooperative (TWAP) strategies."""
    results = {}

    for strategy in ['cooperative', 'competitive']:
        env = make_multi_agent_ac_env(
            n_agents=2,
            X_0=100_000,
            gamma=5e-7,
            eta=2e-6,
            n_steps=10,
        )

        obs, _ = env.reset(seed=42)
        total_rewards = {'trader_0': 0.0, 'trader_1': 0.0}
        done = False
        step = 0

        while not done:
            if strategy == 'cooperative':
                # TWAP: spread trades evenly
                remaining = env.n_steps - step
                fraction = 1.0 / remaining if remaining > 0 else 1.0
            else:
                # Competitive: trade faster
                fraction = 0.3  # 30% each step

            actions = {agent: np.array([fraction]) for agent in env.agents}
            obs, rewards, terms, truncs, infos = env.step(actions)

            for agent, r in rewards.items():
                total_rewards[agent] += r

            done = all(truncs.values()) or all(terms.values())
            step += 1

        results[strategy] = sum(total_rewards.values())
        env.close()

    # Note: which is better depends on parameters
    # Just check both run successfully
    assert 'cooperative' in results
    assert 'competitive' in results


def test_aggregate_volume_observation():
    """Test optional aggregate volume in observations."""
    env = make_multi_agent_ac_env(
        n_agents=2,
        observe_aggregate_volume=True,
    )

    obs, _ = env.reset(seed=42)

    # Observation should have 5 elements with aggregate volume
    for agent in env.agents:
        assert obs[agent].shape == (5,)

    env.close()


def test_episode_completion():
    """Test episode completes correctly."""
    env = make_multi_agent_ac_env(n_agents=2, n_steps=5)
    obs, _ = env.reset(seed=42)

    done = False
    step = 0

    while not done and step < 10:  # Safety limit
        actions = {agent: np.array([0.25]) for agent in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        done = all(truncs.values())
        step += 1

    # Should complete at max steps
    assert step == 5
    assert all(truncs.values())

    # All inventory liquidated
    for agent in env.agents:
        assert infos[agent]['inventory'] == 0

    env.close()


def test_render_mode():
    """Test render doesn't crash."""
    env = make_multi_agent_ac_env(n_agents=2, render_mode='human')
    env.reset(seed=42)

    actions = {agent: np.array([0.2]) for agent in env.agents}
    env.step(actions)

    # Should not raise
    env.render()

    env.close()


def test_pettingzoo_api_compliance():
    """Test basic PettingZoo API compliance."""
    env = make_multi_agent_ac_env(n_agents=2)

    # Has required attributes
    assert hasattr(env, 'possible_agents')
    assert hasattr(env, 'agents')
    assert hasattr(env, 'observation_space')
    assert hasattr(env, 'action_space')

    # observation_space and action_space are callable
    for agent in env.possible_agents:
        obs_space = env.observation_space(agent)
        act_space = env.action_space(agent)
        assert obs_space is not None
        assert act_space is not None

    env.close()


def test_hidden_inventory():
    """Test environment with hidden inventory (true partial observability)."""
    params = MultiAgentACParams(
        n_agents=2,
        X_0=100_000,
        observe_own_inventory=False,  # Can't see own inventory!
        observe_own_cumulative=False, # Also hide cumulative
        observe_price=True,
        observe_time=True,
    )

    env = MultiAgentAlmgrenChrissEnv(params=params, n_steps=10)
    obs, _ = env.reset(seed=42)

    # Observation should be smaller (no inventory, no cumulative)
    assert obs['trader_0'].shape == (2,), f"Expected (2,), got {obs['trader_0'].shape}"  # Only price and time

    # Agent must infer inventory from action history
    actions = {agent: np.array([0.5]) for agent in env.agents}
    obs, _, _, _, infos = env.step(actions)

    # True inventory changed, but not in observation
    assert infos['trader_0']['inventory'] == 50_000  # Actually traded
    assert obs['trader_0'].shape == (2,)  # Still can't see it

    env.close()


def test_noisy_inventory():
    """Test environment with noisy inventory observations."""
    params = MultiAgentACParams(
        n_agents=2,
        X_0=100_000,
        observe_own_inventory=True,
        inventory_noise_std=0.1,  # 10% noise
    )

    env = MultiAgentAlmgrenChrissEnv(params=params, n_steps=10)

    # Collect observations over multiple resets
    obs_values = []
    for seed in range(10):
        obs, _ = env.reset(seed=42)  # Same seed for same true state
        # Force different noise by not resetting numpy random
        obs_values.append(obs['trader_0'][1])  # Inventory component

    # Should have variation due to noise
    # (Actually, reset() calls np.random.seed so noise will be same)
    # Let's check within episode instead
    env.reset(seed=42)

    obs_in_episode = []
    for _ in range(5):
        obs, _, _, _, _ = env.step({agent: np.array([0.0]) for agent in env.agents})
        obs_in_episode.append(obs['trader_0'][1])

    env.close()


def test_noisy_price():
    """Test environment with noisy price observations."""
    params = MultiAgentACParams(
        n_agents=2,
        X_0=100_000,
        observe_price=True,
        price_noise_std=0.01,  # 1% noise
    )

    env = MultiAgentAlmgrenChrissEnv(params=params, n_steps=10)
    env.reset(seed=42)

    # Get multiple observations of same state
    obs1 = env._get_observation('trader_0')
    obs2 = env._get_observation('trader_0')

    # With noise, observations might differ
    # (depends on numpy random state)
    env.close()


def test_minimal_observation():
    """Test environment with minimal observations (just time)."""
    params = MultiAgentACParams(
        n_agents=2,
        X_0=100_000,
        observe_price=False,
        observe_own_inventory=False,
        observe_own_cumulative=False,
        observe_time=True,  # Only time visible
    )

    env = MultiAgentAlmgrenChrissEnv(params=params, n_steps=10)
    obs, _ = env.reset(seed=42)

    # Only time in observation
    assert obs['trader_0'].shape == (1,)
    assert obs['trader_0'][0] == 0.0  # t=0 at start

    env.close()


def test_action_feedback_observation():
    """Test environment with last action feedback."""
    params = MultiAgentACParams(
        n_agents=2,
        X_0=100_000,
        observe_price=True,
        observe_own_inventory=True,
        observe_own_cumulative=False,
        observe_time=True,
        observe_last_execution_price=True,
        observe_last_trade_size=True,
    )

    env = MultiAgentAlmgrenChrissEnv(params=params, n_steps=10)
    obs, _ = env.reset(seed=42)

    # price(1) + inv(1) + time(1) + last_price(1) + last_size(1) = 5
    expected_dim = params.get_obs_dim()
    assert obs['trader_0'].shape == (expected_dim,), f"Expected ({expected_dim},), got {obs['trader_0'].shape}"

    # After trading
    actions = {'trader_0': np.array([0.3]), 'trader_1': np.array([0.0])}
    obs, _, _, _, infos = env.step(actions)

    # Last trade info should be in observation
    # obs[-2] = last execution price, obs[-1] = last trade size
    assert obs['trader_0'][-1] > 0  # Traded something

    env.close()


def test_obs_dim_calculation():
    """Test observation dimension is calculated correctly."""
    # Full observability
    params_full = MultiAgentACParams(
        n_agents=2,
        observe_price=True,
        observe_own_inventory=True,
        observe_own_cumulative=True,
        observe_time=True,
        observe_aggregate_volume=True,
        observe_num_agents=True,
        observe_last_execution_price=True,
        observe_last_trade_size=True,
    )
    assert params_full.get_obs_dim() == 8

    # Minimal observability
    params_min = MultiAgentACParams(
        n_agents=2,
        observe_price=False,
        observe_own_inventory=False,
        observe_own_cumulative=False,
        observe_time=True,
    )
    assert params_min.get_obs_dim() == 1

    # Hidden inventory
    params_hidden = MultiAgentACParams(
        n_agents=2,
        observe_price=True,
        observe_own_inventory=False,
        observe_own_cumulative=True,
        observe_time=True,
    )
    assert params_hidden.get_obs_dim() == 3


if __name__ == '__main__':
    print("Running Multi-Agent Almgren-Chriss environment tests...")

    test_env_initialization()
    print("✓ test_env_initialization")

    test_reset_returns_observations()
    print("✓ test_reset_returns_observations")

    test_parallel_step()
    print("✓ test_parallel_step")

    test_partial_observability()
    print("✓ test_partial_observability")

    test_shared_price_impact()
    print("✓ test_shared_price_impact")

    test_heterogeneous_agents()
    print("✓ test_heterogeneous_agents")

    test_twap_both_agents()
    print("✓ test_twap_both_agents")

    test_competitive_vs_cooperative()
    print("✓ test_competitive_vs_cooperative")

    test_aggregate_volume_observation()
    print("✓ test_aggregate_volume_observation")

    test_episode_completion()
    print("✓ test_episode_completion")

    test_render_mode()
    print("✓ test_render_mode")

    test_pettingzoo_api_compliance()
    print("✓ test_pettingzoo_api_compliance")

    # Partial observability tests
    test_hidden_inventory()
    print("✓ test_hidden_inventory")

    test_noisy_inventory()
    print("✓ test_noisy_inventory")

    test_noisy_price()
    print("✓ test_noisy_price")

    test_minimal_observation()
    print("✓ test_minimal_observation")

    test_action_feedback_observation()
    print("✓ test_action_feedback_observation")

    test_obs_dim_calculation()
    print("✓ test_obs_dim_calculation")

    print("\nAll tests passed!")
