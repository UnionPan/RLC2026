"""Information Set MCTS (ISMCTS) with determinization hook."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .utils import clone_env

def _default_actions(action_space) -> List[int]:
    if hasattr(action_space, "n"):
        return list(range(action_space.n))
    if isinstance(action_space, Sequence):
        return list(action_space)
    raise ValueError("Unsupported action space")


def _step_env(env, action) -> Tuple[Any, float, bool]:
    current_agent = getattr(env, "agent_selection", None)
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


class _Node:
    __slots__ = ("parent", "children", "n", "w")

    def __init__(self, parent: Optional["_Node"] = None):
        self.parent = parent
        self.children: Dict[int, _Node] = {}
        self.n = 0
        self.w = 0.0

    @property
    def q(self) -> float:
        return self.w / self.n if self.n > 0 else 0.0


class ISMCTSPlanner:
    """ISMCTS with determinization via a state sampler."""

    def __init__(
        self,
        action_space,
        n_simulations: int = 200,
        max_depth: int = 50,
        gamma: float = 0.99,
        c_puct: float = 1.4,
        rollout_policy: Optional[Callable[[Any, List[int]], int]] = None,
        state_sampler: Optional[Callable[[Any], Any]] = None,
        observation_key: Optional[Callable[[Any], Any]] = None,
    ):
        self.action_space = action_space
        self.actions = _default_actions(action_space)
        self.n_simulations = n_simulations
        self.max_depth = max_depth
        self.gamma = gamma
        self.c_puct = c_puct
        self.rollout_policy = rollout_policy or self._random_policy
        self.state_sampler = state_sampler or self._default_state_sampler
        self.observation_key = observation_key or _obs_key

        self.root = _Node()
        self.info_sets: Dict[Any, _Node] = {}

    def plan(self, env, observation: Optional[Any] = None) -> int:
        info_key = self.observation_key(observation) if observation is not None else None
        root = self.info_sets.get(info_key, self.root)

        for _ in range(self.n_simulations):
            sim_env = self.state_sampler(env)
            self._simulate(sim_env, root, depth=0)

        if not root.children:
            return self.actions[0]
        return max(root.children.items(), key=lambda kv: kv[1].n)[0]

    def _simulate(self, env, node: _Node, depth: int) -> float:
        if depth >= self.max_depth:
            return 0.0

        if len(node.children) < len(self.actions):
            action = self._expand(node)
            _, reward, done = _step_env(env, action)
            total = reward
            if not done and depth + 1 < self.max_depth:
                total += self.gamma * self._rollout(env, depth + 1)
            self._backup(node.children[action], total)
            return total

        action = self._select(node)
        _, reward, done = _step_env(env, action)
        total = reward
        if not done:
            total += self.gamma * self._simulate(env, node.children[action], depth + 1)
        self._backup(node.children[action], total)
        return total

    def _select(self, node: _Node) -> int:
        best_action = None
        best_score = -float("inf")
        for action in self.actions:
            child = node.children[action]
            ucb = child.q + self.c_puct * math.sqrt(math.log(node.n + 1) / (child.n + 1))
            if ucb > best_score:
                best_score = ucb
                best_action = action
        return best_action

    def _expand(self, node: _Node) -> int:
        for action in self.actions:
            if action not in node.children:
                node.children[action] = _Node(parent=node)
                return action
        return self.actions[0]

    def _rollout(self, env, depth: int) -> float:
        total = 0.0
        discount = 1.0
        for _ in range(depth, self.max_depth):
            action = self.rollout_policy(env, self.actions)
            _, reward, done = _step_env(env, action)
            total += discount * reward
            if done:
                break
            discount *= self.gamma
        return total

    def _backup(self, node: _Node, value: float) -> None:
        while node is not None:
            node.n += 1
            node.w += value
            node = node.parent

    @staticmethod
    def _random_policy(env, actions: List[int]) -> int:
        return actions[np.random.randint(len(actions))]

    @staticmethod
    def _default_state_sampler(env):
        return clone_env(env)
