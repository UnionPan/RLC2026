"""
Tests for Almgren-Chriss Optimal Execution Environment.

Tests:
1. Environment initialization and reset
2. Observation and action spaces
3. Step mechanics and inventory tracking
4. Optimal strategy correctness (closed-form)
5. Comparison of RL agent vs optimal
"""

import os
import sys
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import using full path
from lib.fin.simulations.almgren_chriss_env import (
    AlmgrenChrissEnv,
    AlmgrenChrissParams,
    make_almgren_chriss_env,
)


def test_env_initialization():
    """Test environment initializes correctly."""
    env = make_almgren_chriss_env()

    assert env.params.S_0 == 100.0
    assert env.params.X_0 == 1_000_000
    assert env.n_steps == 20
    assert env.action_space is not None
    assert env.observation_space is not None

    env.close()


def test_reset_and_observation():
    """Test reset returns valid observation."""
    env = make_almgren_chriss_env()
    obs, info = env.reset(seed=42)

    # Observation shape
    assert obs.shape == (3,)

    # Normalized observation bounds
    assert obs[0] > 0  # Price
    assert 0 <= obs[1] <= 1  # Inventory fraction
    assert obs[2] == 0  # Time = 0 at start

    # Info dict
    assert 'price' in info
    assert 'inventory' in info
    assert info['inventory'] == env.params.X_0

    env.close()


def test_step_reduces_inventory():
    """Test that trading reduces inventory."""
    env = make_almgren_chriss_env()
    obs, _ = env.reset(seed=42)

    initial_inventory = env.q

    # Trade 50% of remaining
    action = np.array([0.5])
    obs, reward, terminated, truncated, info = env.step(action)

    # Inventory should decrease
    assert env.q < initial_inventory
    assert env.q == initial_inventory * 0.5

    # Cash should increase
    assert env.cash > 0

    # Time should advance
    assert env.t > 0

    env.close()


def test_full_liquidation():
    """Test liquidating all inventory."""
    env = make_almgren_chriss_env(n_steps=5)
    obs, _ = env.reset(seed=42)

    # Trade 100% each step
    done = False
    step_count = 0
    while not done:
        action = np.array([1.0])  # Trade all remaining
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1

        if step_count > 10:  # Safety
            break

    # Should terminate with no inventory
    assert env.q == 0 or env.q < 1  # Allow small floating point
    assert terminated  # Terminated due to full liquidation

    env.close()


def test_optimal_trajectory_twap():
    """Test TWAP is optimal for risk-neutral agent."""
    # Risk-neutral: lambda_var = 0 → TWAP is optimal
    params = AlmgrenChrissParams(
        S_0=100.0,
        X_0=1000,
        sigma=0.02,
        gamma=0.0,  # No permanent impact
        eta=1e-6,
        T=1.0,
        lambda_var=0.0,  # Risk-neutral
    )
    env = AlmgrenChrissEnv(params=params, n_steps=10)

    # Optimal is TWAP: constant trade rate
    trajectory = env.get_optimal_trajectory()

    # Inventory should decrease linearly
    expected_inventory = params.X_0 * (1 - trajectory['t'] / params.T)
    np.testing.assert_allclose(trajectory['inventory'], expected_inventory, rtol=1e-5)

    env.close()


def test_optimal_trajectory_risk_averse():
    """Test front-loaded execution for risk-averse agent."""
    params = AlmgrenChrissParams(
        S_0=100.0,
        X_0=1000,
        sigma=0.02,
        gamma=0.0,
        eta=1e-6,
        T=1.0,
        lambda_var=1e-4,  # Risk-averse
    )
    env = AlmgrenChrissEnv(params=params, n_steps=10)

    trajectory = env.get_optimal_trajectory()

    # Risk-averse: trade more at the beginning than TWAP
    # Compare inventory at each time point to TWAP
    twap_inventory = params.X_0 * (1 - trajectory['t'] / params.T)

    # At early times, risk-averse inventory < TWAP (traded more)
    # At late times, risk-averse inventory > TWAP (traded less)
    # Check midpoint: should have traded more than TWAP by then
    mid_idx = len(trajectory['t']) // 2

    # Risk-averse should be below TWAP in first half
    assert trajectory['inventory'][mid_idx] < twap_inventory[mid_idx]

    env.close()


def test_twap_agent_performance():
    """Test TWAP agent performance vs optimal."""
    env = make_almgren_chriss_env(
        X_0=10000,
        n_steps=20,
        reward_type='shortfall',
    )

    # Run TWAP strategy: trade 1/remaining_steps each period
    obs, _ = env.reset(seed=42)

    total_reward = 0
    done = False
    step = 0

    while not done:
        # TWAP: trade equal fraction of remaining each step
        remaining_steps = env.n_steps - step
        if remaining_steps > 0:
            fraction = 1.0 / remaining_steps
        else:
            fraction = 1.0

        action = np.array([fraction])
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        step += 1

    # TWAP should achieve reasonable performance (low shortfall)
    final_shortfall_bps = info['shortfall_bps']
    assert abs(final_shortfall_bps) < 100  # Less than 1% shortfall

    env.close()


def test_optimal_following_agent():
    """Test agent following optimal trajectory."""
    env = make_almgren_chriss_env(
        X_0=10000,
        n_steps=20,
        reward_type='shortfall',
    )

    obs, _ = env.reset(seed=42)

    total_reward = 0
    done = False

    while not done:
        # Compute optimal shares to trade this period
        optimal_shares = env.optimal_shares_this_period(env.t, env.dt)
        fraction = optimal_shares / env.q if env.q > 0 else 1.0
        fraction = np.clip(fraction, 0, 1)

        action = np.array([fraction])
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    # Should stay close to optimal trajectory
    assert info['inventory'] < 100  # Nearly fully liquidated

    env.close()


def test_discrete_action_space():
    """Test discrete action space works."""
    env = make_almgren_chriss_env(
        action_type='discrete',
        n_steps=10,
    )

    assert env.action_space.n == 11  # 0%, 10%, ..., 100%

    obs, _ = env.reset(seed=42)

    # Take action index 5 (50%)
    obs, reward, terminated, truncated, info = env.step(5)

    # Should have traded about 50%
    assert env.q < env.params.X_0 * 0.6
    assert env.q > env.params.X_0 * 0.4

    env.close()


def test_market_impact():
    """Test market impact affects execution price."""
    # High impact environment
    high_impact_env = make_almgren_chriss_env(
        gamma=1e-5,  # 10x permanent
        eta=1e-4,    # 10x temporary
        n_steps=10,
    )

    # Low impact environment
    low_impact_env = make_almgren_chriss_env(
        gamma=1e-8,
        eta=1e-7,
        n_steps=10,
    )

    high_impact_env.reset(seed=42)
    low_impact_env.reset(seed=42)

    # Trade same fraction
    high_obs, _, _, _, high_info = high_impact_env.step(np.array([0.5]))
    low_obs, _, _, _, low_info = low_impact_env.step(np.array([0.5]))

    # High impact should have higher shortfall
    assert high_info['shortfall'] > low_info['shortfall']

    high_impact_env.close()
    low_impact_env.close()


def test_expected_cost_formula():
    """Test expected cost matches theoretical formula."""
    params = AlmgrenChrissParams(
        S_0=100.0,
        X_0=1000,
        sigma=0.02,
        gamma=1e-6,
        eta=1e-5,
        T=1.0,
        lambda_var=1e-5,
    )
    env = AlmgrenChrissEnv(params=params, n_steps=100)

    expected_cost, var_cost = env.expected_cost()

    # Sanity checks
    assert expected_cost > 0  # Cost is positive
    assert var_cost > 0  # Variance is positive

    # Permanent cost component
    perm_cost = 0.5 * params.gamma * params.X_0**2
    assert expected_cost >= perm_cost  # Total cost >= permanent cost

    env.close()


def test_render_mode():
    """Test render mode doesn't crash."""
    env = make_almgren_chriss_env(render_mode='human')
    env.reset(seed=42)

    # Should not raise
    env.render()

    env.step(np.array([0.2]))
    env.render()

    env.close()


if __name__ == '__main__':
    print("Running Almgren-Chriss environment tests...")

    test_env_initialization()
    print("✓ test_env_initialization")

    test_reset_and_observation()
    print("✓ test_reset_and_observation")

    test_step_reduces_inventory()
    print("✓ test_step_reduces_inventory")

    test_full_liquidation()
    print("✓ test_full_liquidation")

    test_optimal_trajectory_twap()
    print("✓ test_optimal_trajectory_twap")

    test_optimal_trajectory_risk_averse()
    print("✓ test_optimal_trajectory_risk_averse")

    test_twap_agent_performance()
    print("✓ test_twap_agent_performance")

    test_optimal_following_agent()
    print("✓ test_optimal_following_agent")

    test_discrete_action_space()
    print("✓ test_discrete_action_space")

    test_market_impact()
    print("✓ test_market_impact")

    test_expected_cost_formula()
    print("✓ test_expected_cost_formula")

    test_render_mode()
    print("✓ test_render_mode")

    print("\nAll tests passed!")
