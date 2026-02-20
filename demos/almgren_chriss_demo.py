"""
Almgren-Chriss Optimal Execution Demo.

Focus:
1. Closed-form optimal Almgren-Chriss policy (single-agent optimum)
2. TWAP baseline comparison
3. Monte Carlo comparison under identical environment settings

Usage:
    python demos/almgren_chriss_demo.py --policy optimal --mode human
    python demos/almgren_chriss_demo.py --policy optimal --mode pygame
    python demos/almgren_chriss_demo.py --compare --episodes 30
"""

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lib.fin.agents import OptimalACAgent, TWAPAgent
from lib.fin.simulations import AlmgrenChrissEnv, make_almgren_chriss_env


def make_env(render_mode: Optional[str] = None) -> AlmgrenChrissEnv:
    """Create a standard A-C environment used by this demo."""
    return make_almgren_chriss_env(
        S_0=100.0,
        X_0=100_000,
        sigma=0.02,
        gamma=5e-7,
        eta=2e-6,
        T=1.0,
        n_steps=20,
        lambda_var=1e-6,
        reward_type="shortfall",
        render_mode=render_mode,
    )


def make_agent(policy_name: str, env: AlmgrenChrissEnv):
    """Factory for demo policies."""
    if policy_name == "optimal":
        return OptimalACAgent.from_env(env)
    if policy_name == "twap":
        return TWAPAgent()
    raise ValueError(f"Unknown policy: {policy_name}")


def run_episode(
    env: AlmgrenChrissEnv,
    agent,
    seed: int,
    render: bool,
    sleep_s: float,
) -> Dict[str, Any]:
    """Run one episode with a given policy and return summary metrics."""
    obs, info = env.reset(seed=seed)
    agent.reset()

    total_reward = 0.0
    done = False

    if render:
        env.render()

    while not done:
        action = agent.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        if render:
            frame = env.render()
            if env.render_mode == "pygame" and frame is None:
                break
            if sleep_s > 0:
                time.sleep(sleep_s)

    return {
        "total_reward": float(total_reward),
        "cash": float(info.get("cash", np.nan)),
        "shortfall": float(info.get("shortfall", np.nan)),
        "shortfall_bps": float(info.get("shortfall_bps", np.nan)),
        "steps": int(info.get("step", env.step_count)),
    }


def print_summary(label: str, result: Dict[str, Any]) -> None:
    print(f"\n{label}")
    print("-" * len(label))
    print(f"  Total reward     : {result['total_reward']:.6f}")
    print(f"  Final cash       : ${result['cash']:,.2f}")
    print(f"  Shortfall        : ${result['shortfall']:,.2f}")
    print(f"  Shortfall (bps)  : {result['shortfall_bps']:.2f}")
    print(f"  Steps            : {result['steps']}")


def hold_pygame_final_frame(env: AlmgrenChrissEnv) -> None:
    """Keep pygame window open on final state until user closes it."""
    if env.render_mode != "pygame":
        return

    renderer = getattr(env, "renderer", None)
    wait_until_closed = getattr(renderer, "wait_until_closed", None)
    if callable(wait_until_closed):
        print("\nEpisode finished. Close the pygame window (or press ESC) to exit.")
        wait_until_closed()


def run_single(policy_name: str, mode: str, sleep_s: float) -> None:
    render_mode = None if mode == "none" else mode
    env = make_env(render_mode=render_mode)
    agent = make_agent(policy_name, env)

    expected_cost, var_cost = env.expected_cost()

    print("\nAlmgren-Chriss Single-Agent Demo")
    print("=" * 40)
    print(f"Policy           : {policy_name}")
    print(f"Render mode      : {mode}")
    print(f"Theoretical E[C] : ${expected_cost:,.2f}")
    print(f"Theoretical Std[C]: ${np.sqrt(var_cost):,.2f}")

    result = run_episode(
        env=env,
        agent=agent,
        seed=42,
        render=(mode != "none"),
        sleep_s=sleep_s,
    )
    print_summary(f"Episode Result ({policy_name})", result)

    hold_pygame_final_frame(env)
    env.close()


def compare_policies(n_episodes: int) -> None:
    print("\nAlmgren-Chriss Monte Carlo Comparison")
    print("=" * 44)
    print(f"Episodes: {n_episodes}")

    metrics: Dict[str, Dict[str, List[float]]] = {
        "optimal": {"reward": [], "shortfall_bps": []},
        "twap": {"reward": [], "shortfall_bps": []},
    }

    for seed in range(n_episodes):
        for policy_name in ("optimal", "twap"):
            env = make_env(render_mode=None)
            agent = make_agent(policy_name, env)
            result = run_episode(env, agent, seed=seed, render=False, sleep_s=0.0)
            metrics[policy_name]["reward"].append(result["total_reward"])
            metrics[policy_name]["shortfall_bps"].append(result["shortfall_bps"])
            env.close()

    print("\n{:<10} {:>18} {:>22}".format("Policy", "Mean Reward", "Mean Shortfall (bps)"))
    print("-" * 54)
    for policy_name in ("optimal", "twap"):
        reward_arr = np.array(metrics[policy_name]["reward"], dtype=float)
        sf_arr = np.array(metrics[policy_name]["shortfall_bps"], dtype=float)
        print(
            "{:<10} {:>9.6f} ± {:<8.6f} {:>9.3f} ± {:<8.3f}".format(
                policy_name,
                reward_arr.mean(),
                reward_arr.std(ddof=0),
                sf_arr.mean(),
                sf_arr.std(ddof=0),
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Almgren-Chriss optimal policy demo")
    parser.add_argument(
        "--policy",
        type=str,
        default="optimal",
        choices=["optimal", "twap"],
        help="Policy for single-run mode",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="human",
        choices=["none", "human", "pygame"],
        help="Rendering mode for single-run mode",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run Monte Carlo comparison (optimal vs TWAP)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of episodes for --compare",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.08,
        help="Sleep per step for rendered runs",
    )
    args = parser.parse_args()

    if args.compare:
        compare_policies(n_episodes=args.episodes)
    else:
        run_single(policy_name=args.policy, mode=args.mode, sleep_s=args.sleep)


if __name__ == "__main__":
    main()
