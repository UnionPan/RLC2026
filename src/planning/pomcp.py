"""POMCP planner using a particle belief and simulator rollouts."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .utils import clone_env

def _default_actions(action_space) -> List[int]:
    if hasattr(action_space, "n"):
        return list(range(action_space.n))
    raise ValueError("Unsupported action space")


def _step_env(env, action) -> Tuple[Any, float, bool]:
    try:
        current_agent = env.agent_selection
    except Exception:
        current_agent = None
    result = env.step(action)
    if isinstance(result, tuple) and len(result) >= 4:
        obs, reward, terminated, truncated = result[:4]
        done = bool(terminated) or bool(truncated)
        return obs, float(reward), done
    if isinstance(result, tuple) and len(result) == 3:
        obs, reward, done = result
        return obs, float(reward), bool(done)
    if hasattr(env, "rewards"):
        rewards = env.rewards
        if current_agent is not None:
            reward = float(rewards.get(current_agent, 0.0))
            done = bool(env.terminations.get(current_agent, False) or env.truncations.get(current_agent, False))
        else:
            reward = float(sum(rewards.values())) if isinstance(rewards, dict) else 0.0
            done = bool(any(getattr(env, "terminations", {}).values()) or any(getattr(env, "truncations", {}).values()))
        if hasattr(env, "observe"):
            try:
                obs = env.observe(env.agent_selection)
            except Exception:
                obs = None
        elif hasattr(env, "last"):
            obs = env.last()[0]
        else:
            obs = None
        return obs, reward, done
    raise ValueError("Unsupported env.step signature")


def _obs_key(obs: Any) -> Any:
    if isinstance(obs, np.ndarray):
        return obs.tobytes()
    if isinstance(obs, (list, tuple)):
        return tuple(obs)
    return obs


class _ActionNode:
    __slots__ = ("n", "w", "children")

    def __init__(self):
        self.n = 0
        self.w = 0.0
        self.children: Dict[Any, _ObservationNode] = {}

    @property
    def q(self) -> float:
        return self.w / self.n if self.n > 0 else 0.0


class _ObservationNode:
    __slots__ = ("n", "children")

    def __init__(self):
        self.n = 0
        self.children: Dict[int, _ActionNode] = {}


class POMCPPlanner:
    """POMCP planner with particle belief."""

    def __init__(
        self,
        action_space,
        n_simulations: int = 200,
        max_depth: int = 50,
        gamma: float = 0.99,
        c_puct: float = 1.4,
        rollout_policy: Optional[Callable[[Any, List[int]], int]] = None,
        simulator_step: Optional[Callable[[Any, int], Tuple[Any, float, bool]]] = None,
        max_particles: int = 200,
    ):
        self.action_space = action_space
        self.actions = _default_actions(action_space)
        self.n_simulations = n_simulations
        self.max_depth = max_depth
        self.gamma = gamma
        self.c_puct = c_puct
        self.rollout_policy = rollout_policy or self._random_policy
        self.simulator_step = simulator_step or _step_env
        self.max_particles = max_particles

        self.root = _ObservationNode()
        self.particles: List[Any] = []

    def update_belief(self, env, action: int, observation: Any) -> None:
        """Update particles by filtering on the latest (action, observation)."""
        if not self.particles:
            self.particles = [clone_env(env) for _ in range(self.max_particles)]

        new_particles = []
        obs_key = _obs_key(observation)
        for particle in self.particles:
            _, _, done = self.simulator_step(particle, action)
            if done:
                continue
            obs, _, _ = self.simulator_step(particle, 0)
            if _obs_key(obs) == obs_key:
                new_particles.append(particle)
            if len(new_particles) >= self.max_particles:
                break

        if not new_particles:
            new_particles = [clone_env(env) for _ in range(self.max_particles)]
        self.particles = new_particles

    def plan(self, env) -> int:
        if not self.particles:
            self.particles = [clone_env(env) for _ in range(self.max_particles)]

        for _ in range(self.n_simulations):
            particle = clone_env(self.particles[np.random.randint(len(self.particles))])
            self._simulate(particle, self.root, depth=0)

        if not self.root.children:
            return self.actions[0]
        return max(self.root.children.items(), key=lambda kv: kv[1].n)[0]

    def _simulate(self, env, node: _ObservationNode, depth: int) -> float:
        if depth >= self.max_depth:
            return 0.0

        if len(node.children) < len(self.actions):
            action = self._expand(node)
            obs, reward, done = self.simulator_step(env, action)
            total = reward
            if not done and depth + 1 < self.max_depth:
                total += self.gamma * self._rollout(env, depth + 1)
            self._backup(node, action, obs, total)
            return total

        action = self._select(node)
        obs, reward, done = self.simulator_step(env, action)
        total = reward
        if not done:
            obs_key = _obs_key(obs)
            if obs_key not in node.children[action].children:
                node.children[action].children[obs_key] = _ObservationNode()
            total += self.gamma * self._simulate(env, node.children[action].children[obs_key], depth + 1)
        self._backup(node, action, obs, total)
        return total

    def _select(self, node: _ObservationNode) -> int:
        best_action = None
        best_score = -float("inf")
        for action in self.actions:
            child = node.children[action]
            ucb = child.q + self.c_puct * math.sqrt(math.log(node.n + 1) / (child.n + 1))
            if ucb > best_score:
                best_score = ucb
                best_action = action
        return best_action

    def _expand(self, node: _ObservationNode) -> int:
        for action in self.actions:
            if action not in node.children:
                node.children[action] = _ActionNode()
                return action
        return self.actions[0]

    def _rollout(self, env, depth: int) -> float:
        total = 0.0
        discount = 1.0
        for _ in range(depth, self.max_depth):
            action = self.rollout_policy(env, self.actions)
            _, reward, done = self.simulator_step(env, action)
            total += discount * reward
            if done:
                break
            discount *= self.gamma
        return total

    def _backup(self, node: _ObservationNode, action: int, obs: Any, value: float) -> None:
        node.n += 1
        child = node.children[action]
        child.n += 1
        child.w += value
        obs_key = _obs_key(obs)
        if obs_key not in child.children:
            child.children[obs_key] = _ObservationNode()

    @staticmethod
    def _random_policy(env, actions: List[int]) -> int:
        return actions[np.random.randint(len(actions))]
