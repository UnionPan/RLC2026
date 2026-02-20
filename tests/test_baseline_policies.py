"""
Tests for Baseline Execution Policies.

Tests:
1. TWAP agent correctness
2. Optimal A-C agent matches env method
3. Multi-agent policy wrapper
4. Agent reset behavior

Author: Yunian Pan
Email: yp1170@nyu.edu
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lib.fin.agents import (
    TWAPAgent,
    VWAPAgent,
    OptimalACAgent,
    AggressiveAgent,
    ConservativeAgent,
    RandomAgent,
    ConstantFractionAgent,
    AdaptiveAgent,
    MultiAgentPolicy,
    get_optimal_agent_for_env,
)
from lib.fin.simulations import (
    AlmgrenChrissEnv,
    AlmgrenChrissParams,
    make_almgren_chriss_env,
    MultiAgentAlmgrenChrissEnv,
    make_multi_agent_ac_env,
)


def test_twap_agent():
    """Test TWAP agent produces correct fractions."""
    agent = TWAPAgent()

    # 10 steps remaining -> fraction = 0.1
    obs = np.array([1.0, 1.0, 0.0, 0.0])
    info = {'remaining_steps': 10}
    action = agent.act(obs, info)
    assert abs(action[0] - 0.1) < 1e-6

    # 1 step remaining -> fraction = 1.0
    info = {'remaining_steps': 1}
    action = agent.act(obs, info)
    assert abs(action[0] - 1.0) < 1e-6

    # 0 steps remaining -> fraction = 1.0 (edge case)
    info = {'remaining_steps': 0}
    action = agent.act(obs, info)
    assert abs(action[0] - 1.0) < 1e-6


def test_twap_in_env():
    """Test TWAP agent in actual environment."""
    env = make_almgren_chriss_env(X_0=100_000, n_steps=10)
    agent = TWAPAgent()

    obs, info = env.reset(seed=42)
    total_reward = 0
    done = False

    while not done:
        action = agent.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    # Should complete successfully
    assert info['inventory'] == 0
    env.close()


def test_optimal_agent_matches_env():
    """Test optimal agent matches environment's optimal method."""
    params = AlmgrenChrissParams(
        X_0=100_000,
        sigma=0.02,
        eta=2e-6,
        lambda_var=1e-6,
        T=1.0,
    )
    env = AlmgrenChrissEnv(params=params, n_steps=10)

    # Create agent from env
    agent = OptimalACAgent.from_env(env)

    # Compare optimal inventory at various times
    for t in [0.0, 0.25, 0.5, 0.75]:
        env_optimal = env.optimal_inventory(t)
        agent_optimal = agent.optimal_inventory(t)
        assert abs(env_optimal - agent_optimal) < 1e-6, f"Mismatch at t={t}"

    env.close()


def test_optimal_agent_in_env():
    """Test optimal agent executes correctly."""
    env = make_almgren_chriss_env(X_0=100_000, n_steps=20)
    agent = OptimalACAgent.from_env(env)

    obs, info = env.reset(seed=42)
    total_reward = 0
    done = False

    while not done:
        action = agent.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    # Should complete successfully
    assert info['inventory'] == 0
    env.close()


def test_aggressive_agent():
    """Test aggressive agent front-loads trades."""
    agent = AggressiveAgent(early_fraction=0.4, late_fraction=0.1, switch_step=3)

    obs = np.array([1.0, 1.0, 0.0, 0.0])
    info = {}

    # First 3 steps should be aggressive
    for i in range(3):
        action = agent.act(obs, info)
        assert abs(action[0] - 0.4) < 1e-6, f"Step {i} should be aggressive"

    # After switch, should be conservative
    for i in range(3):
        action = agent.act(obs, info)
        assert abs(action[0] - 0.1) < 1e-6, f"Step {i+3} should be conservative"

    # Test reset
    agent.reset()
    action = agent.act(obs, info)
    assert abs(action[0] - 0.4) < 1e-6, "After reset should be aggressive again"


def test_conservative_agent():
    """Test conservative agent back-loads trades."""
    agent = ConservativeAgent(base_fraction=0.05, acceleration=2.0)

    obs = np.array([1.0, 1.0, 0.0, 0.0])
    info = {}

    # Should accelerate
    actions = [agent.act(obs, info)[0] for _ in range(5)]

    # Each action should be larger than previous
    for i in range(1, len(actions)):
        assert actions[i] > actions[i-1], f"Action {i} should be larger than {i-1}"


def test_random_agent_reproducibility():
    """Test random agent is reproducible with seed."""
    agent1 = RandomAgent(min_fraction=0.0, max_fraction=0.3, seed=42)
    agent2 = RandomAgent(min_fraction=0.0, max_fraction=0.3, seed=42)

    obs = np.array([1.0])
    info = {}

    for _ in range(10):
        a1 = agent1.act(obs, info)
        a2 = agent2.act(obs, info)
        assert abs(a1[0] - a2[0]) < 1e-10


def test_constant_fraction_agent():
    """Test constant fraction agent."""
    agent = ConstantFractionAgent(fraction=0.15)

    obs = np.array([1.0])
    info = {}

    for _ in range(10):
        action = agent.act(obs, info)
        assert abs(action[0] - 0.15) < 1e-6


def test_adaptive_agent():
    """Test adaptive agent responds to price changes."""
    agent = AdaptiveAgent(base_fraction=0.1, sensitivity=2.0)

    info = {}

    # First observation - baseline
    obs1 = np.array([1.0])
    action1 = agent.act(obs1, info)
    assert abs(action1[0] - 0.1) < 1e-6

    # Price dropped 10% -> should trade more
    obs2 = np.array([0.9])
    action2 = agent.act(obs2, info)
    assert action2[0] > action1[0], "Should trade more when price drops"

    # Reset and test price increase
    agent.reset()
    obs1 = np.array([1.0])
    agent.act(obs1, info)  # Initialize

    # Price rose 10% -> should trade less
    obs2 = np.array([1.1])
    action2 = agent.act(obs2, info)
    assert action2[0] < 0.1, "Should trade less when price rises"


def test_multi_agent_policy():
    """Test multi-agent policy wrapper."""
    env = make_multi_agent_ac_env(n_agents=2, n_steps=10)

    # Create mixed policy
    policy = MultiAgentPolicy({
        'trader_0': TWAPAgent(),
        'trader_1': AggressiveAgent(),
    })

    obs, infos = env.reset(seed=42)
    done = False

    while not done:
        actions = policy.act(obs, infos)

        assert 'trader_0' in actions
        assert 'trader_1' in actions

        obs, rewards, terms, truncs, infos = env.step(actions)
        done = all(truncs.values())

    env.close()


def test_multi_agent_policy_all_twap():
    """Test convenience constructor for all-TWAP policy."""
    agent_names = ['trader_0', 'trader_1', 'trader_2']
    policy = MultiAgentPolicy.all_twap(agent_names)

    assert len(policy.policies) == 3
    for name in agent_names:
        assert isinstance(policy.policies[name], TWAPAgent)


def test_vwap_agent_default():
    """Test VWAP agent with default (uniform) profile."""
    agent = VWAPAgent()

    obs = np.array([1.0])
    info = {'remaining_steps': 5}

    # Without profile, should behave like TWAP
    action = agent.act(obs, info)
    assert abs(action[0] - 0.2) < 1e-6


def test_vwap_agent_with_profile():
    """Test VWAP agent with custom volume profile."""
    # U-shaped volume profile (high at open/close)
    profile = np.array([0.3, 0.1, 0.1, 0.1, 0.1, 0.3])
    agent = VWAPAgent(volume_profile=profile)

    obs = np.array([1.0])
    info = {}

    actions = [agent.act(obs, info)[0] for _ in range(6)]

    assert abs(actions[0] - 0.3) < 1e-6  # High at open
    assert abs(actions[2] - 0.1) < 1e-6  # Low midday
    assert abs(actions[5] - 0.3) < 1e-6  # High at close


def test_get_optimal_agent_for_env():
    """Test utility function creates correct agent."""
    env = make_almgren_chriss_env(X_0=100_000, n_steps=10)
    agent = get_optimal_agent_for_env(env)

    assert isinstance(agent, OptimalACAgent)
    assert abs(agent.kappa - env.kappa) < 1e-10

    env.close()


def test_agent_with_partial_observability():
    """Test agents work with partial observability."""
    from lib.fin.simulations.multi_agent_ac_env import MultiAgentACParams

    params = MultiAgentACParams(
        n_agents=2,
        X_0=100_000,
        observe_own_inventory=False,  # Hidden inventory
        observe_time=True,
    )

    env = MultiAgentAlmgrenChrissEnv(params=params, n_steps=10)

    # TWAP should still work (uses info dict)
    policy = MultiAgentPolicy.all_twap(env.possible_agents)

    obs, infos = env.reset(seed=42)
    done = False

    while not done:
        actions = policy.act(obs, infos)
        obs, rewards, terms, truncs, infos = env.step(actions)
        done = all(truncs.values())

    env.close()


if __name__ == '__main__':
    print("Running Baseline Policy tests...")

    test_twap_agent()
    print("✓ test_twap_agent")

    test_twap_in_env()
    print("✓ test_twap_in_env")

    test_optimal_agent_matches_env()
    print("✓ test_optimal_agent_matches_env")

    test_optimal_agent_in_env()
    print("✓ test_optimal_agent_in_env")

    test_aggressive_agent()
    print("✓ test_aggressive_agent")

    test_conservative_agent()
    print("✓ test_conservative_agent")

    test_random_agent_reproducibility()
    print("✓ test_random_agent_reproducibility")

    test_constant_fraction_agent()
    print("✓ test_constant_fraction_agent")

    test_adaptive_agent()
    print("✓ test_adaptive_agent")

    test_multi_agent_policy()
    print("✓ test_multi_agent_policy")

    test_multi_agent_policy_all_twap()
    print("✓ test_multi_agent_policy_all_twap")

    test_vwap_agent_default()
    print("✓ test_vwap_agent_default")

    test_vwap_agent_with_profile()
    print("✓ test_vwap_agent_with_profile")

    test_get_optimal_agent_for_env()
    print("✓ test_get_optimal_agent_for_env")

    test_agent_with_partial_observability()
    print("✓ test_agent_with_partial_observability")

    print("\nAll tests passed!")
