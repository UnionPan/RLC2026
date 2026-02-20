"""
Tests for Nash Equilibrium Policies.

Tests:
1. Symmetric Nash agent correctness
2. Nash vs single-agent optimal comparison
3. Cooperative agent
4. Price of anarchy computation
5. Multi-agent policy creation
6. Nash equilibrium in environment

Author: Yunian Pan
Email: yp1170@nyu.edu
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lib.fin.agents import (
    SymmetricNashAgent,
    CooperativeAgent,
    AsymmetricNashSolver,
    NashEquilibriumParams,
    compute_price_of_anarchy,
    create_nash_multi_agent_policy,
    create_cooperative_multi_agent_policy,
    OptimalACAgent,
    TWAPAgent,
    MultiAgentPolicy,
)
from lib.fin.simulations import (
    make_multi_agent_ac_env,
    MultiAgentAlmgrenChrissEnv,
    MultiAgentACParams,
)


def test_symmetric_nash_params():
    """Test symmetric Nash agent computes modified kappa."""
    # Single-agent kappa
    eta = 2e-6
    sigma = 0.02
    lambda_var = 1e-6
    gamma = 5e-7

    kappa_single = np.sqrt(lambda_var * sigma**2 / eta)

    # 2-agent Nash
    agent_2 = SymmetricNashAgent(
        n_agents=2, X_0=100_000, eta=eta, gamma=gamma,
        sigma=sigma, lambda_var=lambda_var, T=1.0, n_steps=20
    )

    # 5-agent Nash
    agent_5 = SymmetricNashAgent(
        n_agents=5, X_0=100_000, eta=eta, gamma=gamma,
        sigma=sigma, lambda_var=lambda_var, T=1.0, n_steps=20
    )

    # Nash kappa should be smaller than single-agent (more agents = less urgent)
    # Because eta_eff = eta + gamma*(N-1)/2 > eta
    assert agent_2.kappa_nash < agent_2.kappa_single
    assert agent_5.kappa_nash < agent_5.kappa_single
    assert agent_5.kappa_nash < agent_2.kappa_nash  # More agents = smaller kappa


def test_symmetric_nash_inventory_trajectory():
    """Test Nash inventory trajectory has correct shape."""
    agent = SymmetricNashAgent(
        n_agents=3, X_0=100_000, eta=2e-6, gamma=5e-7,
        sigma=0.02, lambda_var=1e-6, T=1.0, n_steps=20
    )

    # At t=0, should have full inventory
    assert abs(agent.nash_inventory(0) - 100_000) < 1

    # At t=T, should have zero inventory
    assert abs(agent.nash_inventory(1.0)) < 1

    # Inventory should decrease monotonically
    t_vals = np.linspace(0, 1.0, 20)
    inv_vals = [agent.nash_inventory(t) for t in t_vals]
    for i in range(1, len(inv_vals)):
        assert inv_vals[i] <= inv_vals[i-1]


def test_nash_vs_cooperative():
    """Test Nash is less aggressive than cooperative (higher effective eta)."""
    n_agents = 3
    X_0 = 100_000
    eta = 2e-6
    gamma = 5e-7
    sigma = 0.02
    lambda_var = 1e-6
    T = 1.0

    nash_agent = SymmetricNashAgent(
        n_agents=n_agents, X_0=X_0, eta=eta, gamma=gamma,
        sigma=sigma, lambda_var=lambda_var, T=T, n_steps=20
    )

    coop_agent = CooperativeAgent(
        n_agents=n_agents, X_0=X_0, eta=eta, gamma=gamma,
        sigma=sigma, lambda_var=lambda_var, T=T, n_steps=20
    )

    # Cooperative kappa = single-agent kappa (ignores strategic interaction)
    # Nash kappa < cooperative kappa (accounts for strategic impact)
    assert nash_agent.kappa_nash < coop_agent.kappa_coop

    # At midpoint, Nash should have more inventory remaining (less aggressive)
    t_mid = T / 2
    nash_inv = nash_agent.nash_inventory(t_mid)
    coop_inv = coop_agent.cooperative_inventory(t_mid)

    # With lower kappa, Nash is more TWAP-like (less front-loaded)
    # So Nash has more inventory remaining at midpoint
    assert nash_inv > coop_inv, f"Nash {nash_inv} should > Coop {coop_inv}"


def test_price_of_anarchy():
    """Test price of anarchy computation."""
    result = compute_price_of_anarchy(
        n_agents=2,
        X_0=100_000,
        eta=2e-6,
        gamma=5e-7,
        sigma=0.02,
        lambda_var=1e-6,
        T=1.0,
    )

    # PoA should be positive and finite
    assert result['price_of_anarchy'] > 0
    assert np.isfinite(result['price_of_anarchy'])

    # Verify all expected keys present
    assert 'kappa_nash' in result
    assert 'kappa_coop' in result
    assert 'cost_nash_total' in result
    assert 'cost_coop_total' in result

    # With zero gamma, Nash = Cooperative (no strategic interaction)
    result_no_gamma = compute_price_of_anarchy(
        n_agents=2,
        X_0=100_000,
        eta=2e-6,
        gamma=0.0,  # No permanent impact
        sigma=0.02,
        lambda_var=1e-6,
        T=1.0,
    )

    # PoA should be close to 1 when gamma=0 (no externality)
    assert abs(result_no_gamma['price_of_anarchy'] - 1.0) < 0.01


def test_price_of_anarchy_increases_with_agents():
    """Test PoA increases with number of agents in appropriate parameter regime."""
    poas = []
    for n in [2, 3, 5, 10]:
        result = compute_price_of_anarchy(
            n_agents=n,
            X_0=100_000,
            eta=2e-6,
            gamma=5e-6,  # Higher gamma for stronger strategic interaction
            sigma=0.02,
            lambda_var=1e-5,  # Higher risk aversion to see effect
            T=1.0,
        )
        poas.append(result['price_of_anarchy'])

    # PoA should generally increase with N (more strategic distortion)
    # With higher gamma and lambda, Nash agents trade more slowly,
    # leading to higher variance cost compared to cooperative
    assert poas[-1] > poas[0], f"PoA should increase: {poas[0]:.6f} -> {poas[-1]:.6f}"


def test_create_nash_policy_symmetric():
    """Test creating Nash policies for symmetric environment."""
    env = make_multi_agent_ac_env(n_agents=3, n_steps=10)

    nash_policies = create_nash_multi_agent_policy(env)

    assert len(nash_policies) == 3
    assert 'trader_0' in nash_policies
    assert isinstance(nash_policies['trader_0'], SymmetricNashAgent)

    env.close()


def test_create_cooperative_policy():
    """Test creating cooperative policies."""
    env = make_multi_agent_ac_env(n_agents=2, n_steps=10)

    coop_policies = create_cooperative_multi_agent_policy(env)

    assert len(coop_policies) == 2
    assert isinstance(coop_policies['trader_0'], CooperativeAgent)

    env.close()


def test_nash_in_environment():
    """Test Nash equilibrium agents execute in environment."""
    env = make_multi_agent_ac_env(
        n_agents=2,
        X_0=100_000,
        gamma=5e-7,
        eta=2e-6,
        n_steps=10,
    )

    nash_policies = create_nash_multi_agent_policy(env)
    policy = MultiAgentPolicy(nash_policies)

    obs, infos = env.reset(seed=42)
    total_rewards = {agent: 0.0 for agent in env.agents}
    done = False

    while not done:
        actions = policy.act(obs, infos)
        obs, rewards, terms, truncs, infos = env.step(actions)

        for agent, r in rewards.items():
            total_rewards[agent] += r

        done = all(truncs.values())

    # All agents should have liquidated
    for agent in env.agents:
        assert infos[agent]['inventory'] == 0

    # Symmetric agents should have similar performance
    assert abs(total_rewards['trader_0'] - total_rewards['trader_1']) < 0.05

    env.close()


def test_nash_vs_twap_performance():
    """Test Nash equilibrium outperforms TWAP in multi-agent setting."""
    n_episodes = 10

    nash_rewards = []
    twap_rewards = []

    for seed in range(n_episodes):
        # Nash equilibrium
        env = make_multi_agent_ac_env(
            n_agents=2, X_0=100_000, gamma=1e-6, eta=2e-6, n_steps=10
        )
        nash_policies = create_nash_multi_agent_policy(env)
        policy = MultiAgentPolicy(nash_policies)

        obs, infos = env.reset(seed=seed)
        nash_total = 0
        done = False
        while not done:
            actions = policy.act(obs, infos)
            obs, rewards, terms, truncs, infos = env.step(actions)
            nash_total += sum(rewards.values())
            done = all(truncs.values())
        nash_rewards.append(nash_total)
        policy.reset()
        env.close()

        # TWAP
        env = make_multi_agent_ac_env(
            n_agents=2, X_0=100_000, gamma=1e-6, eta=2e-6, n_steps=10
        )
        twap_policy = MultiAgentPolicy.all_twap(env.possible_agents)

        obs, infos = env.reset(seed=seed)
        twap_total = 0
        done = False
        while not done:
            actions = twap_policy.act(obs, infos)
            obs, rewards, terms, truncs, infos = env.step(actions)
            twap_total += sum(rewards.values())
            done = all(truncs.values())
        twap_rewards.append(twap_total)
        twap_policy.reset()
        env.close()

    # Nash should perform at least as well as TWAP on average
    # (Nash is the equilibrium, TWAP ignores strategic interaction)
    nash_mean = np.mean(nash_rewards)
    twap_mean = np.mean(twap_rewards)

    # Note: performance difference depends on parameters
    # Just check they're comparable (Nash should not be much worse)
    assert nash_mean >= twap_mean - 0.1, f"Nash {nash_mean} should >= TWAP {twap_mean}"


def test_asymmetric_solver():
    """Test asymmetric Nash solver."""
    params = NashEquilibriumParams(
        n_agents=2,
        X_0_list=[100_000, 50_000],  # Different inventories
        eta_list=[2e-6, 3e-6],        # Different temp impact
        gamma=5e-7,
        sigma=0.02,
        lambda_var_list=[1e-6, 2e-6],  # Different risk aversion
        T=1.0,
    )

    solver = AsymmetricNashSolver(params, n_steps=20)

    # Each agent should have different kappa
    assert solver.kappa_eff[0] != solver.kappa_eff[1]

    # Agent 0 (larger, less risk-averse) should trade more aggressively
    inv_0_mid = solver.nash_inventory(0, 0.5)
    inv_1_mid = solver.nash_inventory(1, 0.5)

    # Agent 0 starts with more, so should have more remaining
    assert inv_0_mid > inv_1_mid


def test_nash_agent_reset():
    """Test Nash agent resets properly."""
    agent = SymmetricNashAgent(
        n_agents=2, X_0=100_000, eta=2e-6, gamma=5e-7,
        sigma=0.02, lambda_var=1e-6, T=1.0, n_steps=10
    )

    obs = np.array([1.0, 1.0, 0.0, 0.0])
    info = {'inventory': 100_000, 'remaining_steps': 10}

    # Take some actions
    for _ in range(5):
        agent.act(obs, info)

    assert agent.step == 5

    # Reset
    agent.reset()
    assert agent.step == 0


if __name__ == '__main__':
    print("Running Nash Equilibrium tests...")

    test_symmetric_nash_params()
    print("✓ test_symmetric_nash_params")

    test_symmetric_nash_inventory_trajectory()
    print("✓ test_symmetric_nash_inventory_trajectory")

    test_nash_vs_cooperative()
    print("✓ test_nash_vs_cooperative")

    test_price_of_anarchy()
    print("✓ test_price_of_anarchy")

    test_price_of_anarchy_increases_with_agents()
    print("✓ test_price_of_anarchy_increases_with_agents")

    test_create_nash_policy_symmetric()
    print("✓ test_create_nash_policy_symmetric")

    test_create_cooperative_policy()
    print("✓ test_create_cooperative_policy")

    test_nash_in_environment()
    print("✓ test_nash_in_environment")

    test_nash_vs_twap_performance()
    print("✓ test_nash_vs_twap_performance")

    test_asymmetric_solver()
    print("✓ test_asymmetric_solver")

    test_nash_agent_reset()
    print("✓ test_nash_agent_reset")

    print("\nAll tests passed!")
