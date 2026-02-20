"""
Multi-Agent Almgren-Chriss Equilibrium Demo.

Focus:
1. Symmetric Nash equilibrium policy
2. Cooperative (social planner) benchmark
3. Asymmetric Nash policy (heterogeneous agents)
4. Full-observability equilibrium policy used in a partial-observability env

Usage:
    python demos/multi_agent_ac_demo.py --scenario symmetric --policy nash --mode human
    python demos/multi_agent_ac_demo.py --scenario asymmetric --policy nash --mode human
    python demos/multi_agent_ac_demo.py --scenario partial_assumption --mode human
    python demos/multi_agent_ac_demo.py --scenario compare --episodes 30
    python demos/multi_agent_ac_demo.py --scenario pygame --policy nash
"""

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lib.fin.agents import (
    MultiAgentPolicy,
    TWAPAgent,
    create_cooperative_multi_agent_policy,
    create_nash_multi_agent_policy,
)
from lib.fin.simulations import (
    MultiAgentACParams,
    MultiAgentAlmgrenChrissEnv,
    make_multi_agent_ac_env,
)


def make_symmetric_env(
    render_mode: Optional[str] = None,
    info_mode: str = "full",
) -> MultiAgentAlmgrenChrissEnv:
    """Create a symmetric multi-agent A-C environment."""
    return make_multi_agent_ac_env(
        n_agents=2,
        X_0=500_000,
        sigma=0.02,
        gamma=5e-7,
        eta=2e-6,
        T=1.0,
        n_steps=15,
        reward_type="shortfall",
        render_mode=render_mode,
        info_mode=info_mode,
    )


def make_asymmetric_env(
    render_mode: Optional[str] = None,
    info_mode: str = "full",
) -> MultiAgentAlmgrenChrissEnv:
    """Create an asymmetric multi-agent A-C environment."""
    params = MultiAgentACParams(
        n_agents=2,
        X_0_list=[1_000_000, 200_000],
        eta_list=[2e-6, 4e-6],
        lambda_var_list=[1e-6, 3e-6],
        gamma=7.5e-7,
        sigma=0.02,
        T=1.0,
    )
    return MultiAgentAlmgrenChrissEnv(
        params=params,
        n_steps=15,
        reward_type="shortfall",
        normalize_obs=True,
        normalize_reward=True,
        render_mode=render_mode,
        info_mode=info_mode,
    )


def make_partial_observable_env(
    render_mode: Optional[str] = None,
    info_mode: str = "full",
) -> MultiAgentAlmgrenChrissEnv:
    """
    Create a partial-observability env.

    For this demo we keep info_mode configurable so we can run full-info
    equilibrium policies in a PO environment under the full-observability
    assumption.
    """
    params = MultiAgentACParams(
        n_agents=2,
        X_0=400_000,
        sigma=0.02,
        gamma=5e-7,
        eta=2e-6,
        T=1.0,
        observe_price=True,
        observe_own_inventory=False,
        observe_own_cumulative=False,
        observe_time=True,
        observe_aggregate_volume=True,
    )
    return MultiAgentAlmgrenChrissEnv(
        params=params,
        n_steps=15,
        reward_type="shortfall",
        normalize_obs=True,
        normalize_reward=True,
        render_mode=render_mode,
        info_mode=info_mode,
    )


def make_policy(policy_name: str, env: MultiAgentAlmgrenChrissEnv) -> MultiAgentPolicy:
    """Factory for policy profiles."""
    if policy_name == "nash":
        return MultiAgentPolicy(create_nash_multi_agent_policy(env))
    if policy_name == "cooperative":
        return MultiAgentPolicy(create_cooperative_multi_agent_policy(env))
    if policy_name == "twap":
        return MultiAgentPolicy({name: TWAPAgent() for name in env.possible_agents})
    raise ValueError(f"Unknown policy: {policy_name}")


def run_episode(
    env: MultiAgentAlmgrenChrissEnv,
    policy: MultiAgentPolicy,
    seed: int,
    render: bool,
    sleep_s: float,
) -> Dict[str, Any]:
    """Run one multi-agent episode and return metrics."""
    observations, infos = env.reset(seed=seed)
    policy.reset()

    total_rewards = {agent: 0.0 for agent in env.possible_agents}
    done = False

    if render:
        env.render()

    while not done:
        actions = policy.act(observations, infos)
        observations, rewards, terms, truncs, infos = env.step(actions)

        for agent, reward in rewards.items():
            total_rewards[agent] += reward

        done = all(truncs.values()) or all(terms.values())

        if render:
            frame = env.render()
            if env.render_mode == "pygame" and frame is None:
                break
            if sleep_s > 0:
                time.sleep(sleep_s)

    social_reward = float(sum(total_rewards.values()))
    mean_shortfall_bps = float(np.mean([infos[a].get("shortfall_bps", np.nan) for a in infos]))

    return {
        "total_rewards": total_rewards,
        "infos": infos,
        "social_reward": social_reward,
        "mean_shortfall_bps": mean_shortfall_bps,
    }


def print_episode_summary(title: str, result: Dict[str, Any]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(f"  Social reward       : {result['social_reward']:.6f}")
    print(f"  Mean shortfall (bps): {result['mean_shortfall_bps']:.3f}")
    for agent, rew in result["total_rewards"].items():
        info = result["infos"][agent]
        print(
            f"  {agent}: reward={rew:.6f}, shortfall_bps={info.get('shortfall_bps', np.nan):.3f}, "
            f"inventory={info.get('inventory', np.nan):.2f}"
        )


def hold_pygame_final_frame(env: MultiAgentAlmgrenChrissEnv) -> None:
    """Keep pygame window open on final state until user closes it."""
    if env.render_mode != "pygame":
        return

    renderer = getattr(env, "renderer", None)
    wait_until_closed = getattr(renderer, "wait_until_closed", None)
    if callable(wait_until_closed):
        print("\nEpisode finished. Close the pygame window (or press ESC) to exit.")
        wait_until_closed()


def run_symmetric(policy_name: str, mode: str, sleep_s: float) -> None:
    render_mode = None if mode == "none" else mode
    env = make_symmetric_env(render_mode=render_mode, info_mode="full")
    policy = make_policy(policy_name, env)

    print("\nSymmetric Equilibrium Demo")
    print("=" * 36)
    print(f"Policy      : {policy_name}")
    print(f"Render mode : {mode}")

    result = run_episode(env, policy, seed=42, render=(mode != "none"), sleep_s=sleep_s)
    print_episode_summary(f"Result ({policy_name})", result)
    hold_pygame_final_frame(env)
    env.close()


def run_asymmetric(policy_name: str, mode: str, sleep_s: float) -> None:
    if policy_name == "cooperative":
        raise ValueError("Cooperative policy demo is only configured for symmetric envs.")

    render_mode = None if mode == "none" else mode
    env = make_asymmetric_env(render_mode=render_mode, info_mode="full")
    policy = make_policy(policy_name, env)

    print("\nAsymmetric Equilibrium Demo")
    print("=" * 37)
    print(f"Policy      : {policy_name}")
    print(f"Render mode : {mode}")

    result = run_episode(env, policy, seed=42, render=(mode != "none"), sleep_s=sleep_s)
    print_episode_summary(f"Result ({policy_name}, asymmetric)", result)
    hold_pygame_final_frame(env)
    env.close()


def run_partial_assumption(mode: str, sleep_s: float) -> None:
    render_mode = None if mode == "none" else mode

    print("\nPartial-Observability Env + Full-Info Equilibrium Policy")
    print("=" * 58)
    print("This run uses a PO observation model but feeds full info to policies")
    print("(i.e., equilibrium policy under full-observability assumption).")

    env = make_partial_observable_env(render_mode=render_mode, info_mode="full")
    policy = make_policy("nash", env)

    result = run_episode(env, policy, seed=42, render=(mode != "none"), sleep_s=sleep_s)
    print_episode_summary("Result (nash under full-info assumption)", result)
    hold_pygame_final_frame(env)
    env.close()


def compare_symmetric(n_episodes: int) -> None:
    print("\nSymmetric Policy Comparison")
    print("=" * 31)
    print(f"Episodes: {n_episodes}")

    metrics: Dict[str, Dict[str, List[float]]] = {
        "nash": {"social_reward": [], "mean_shortfall_bps": []},
        "cooperative": {"social_reward": [], "mean_shortfall_bps": []},
        "twap": {"social_reward": [], "mean_shortfall_bps": []},
    }

    for seed in range(n_episodes):
        for policy_name in ("nash", "cooperative", "twap"):
            env = make_symmetric_env(render_mode=None, info_mode="full")
            policy = make_policy(policy_name, env)
            result = run_episode(env, policy, seed=seed, render=False, sleep_s=0.0)
            metrics[policy_name]["social_reward"].append(result["social_reward"])
            metrics[policy_name]["mean_shortfall_bps"].append(result["mean_shortfall_bps"])
            env.close()

    print("\n{:<12} {:>18} {:>24}".format("Policy", "Mean Social Reward", "Mean Shortfall (bps)"))
    print("-" * 58)
    for policy_name in ("nash", "cooperative", "twap"):
        sr = np.array(metrics[policy_name]["social_reward"], dtype=float)
        sf = np.array(metrics[policy_name]["mean_shortfall_bps"], dtype=float)
        print(
            "{:<12} {:>9.6f} ± {:<8.6f} {:>10.3f} ± {:<8.3f}".format(
                policy_name,
                sr.mean(),
                sr.std(ddof=0),
                sf.mean(),
                sf.std(ddof=0),
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-agent A-C equilibrium demo")
    parser.add_argument(
        "--scenario",
        type=str,
        default="symmetric",
        choices=["symmetric", "asymmetric", "partial_assumption", "compare", "pygame"],
        help="Demo scenario",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="nash",
        choices=["nash", "cooperative", "twap"],
        help="Policy for single-run scenarios",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="human",
        choices=["none", "human", "pygame"],
        help="Rendering mode for single-run scenarios",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of episodes for comparison scenario",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.10,
        help="Sleep per step for rendered runs",
    )
    args = parser.parse_args()

    if args.scenario == "compare":
        compare_symmetric(n_episodes=args.episodes)
        return

    if args.scenario == "pygame":
        run_symmetric(policy_name=args.policy, mode="pygame", sleep_s=args.sleep)
        return

    if args.scenario == "symmetric":
        run_symmetric(policy_name=args.policy, mode=args.mode, sleep_s=args.sleep)
    elif args.scenario == "asymmetric":
        run_asymmetric(policy_name=args.policy, mode=args.mode, sleep_s=args.sleep)
    elif args.scenario == "partial_assumption":
        run_partial_assumption(mode=args.mode, sleep_s=args.sleep)


if __name__ == "__main__":
    main()
