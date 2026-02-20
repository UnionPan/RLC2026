"""
Wrappers for Almgren-Chriss environments.

Provides:
- DiscretizeActionWrapper: Convert continuous action space to discrete bins
- NormalizeObservationWrapper: Normalize observations for RL training

Author: Yunian Pan
Email: yp1170@nyu.edu
"""

from __future__ import annotations

import functools
import numpy as np
from typing import Any, Dict, Optional, Tuple

from gymnasium import spaces

try:
    from pettingzoo import ParallelEnv
    PETTINGZOO_AVAILABLE = True
except ImportError:
    ParallelEnv = object
    PETTINGZOO_AVAILABLE = False


class DiscretizeActionWrapper(ParallelEnv):
    """
    Wrapper that discretizes continuous action space for multi-agent environments.

    Converts discrete action indices to continuous fractions in [0, 1].
    Useful for applying discrete policy gradient methods (REINFORCE, PPO) to
    environments with continuous action spaces like Almgren-Chriss.

    Example:
        n_bins=21 creates actions 0-20 mapping to fractions 0.0, 0.05, ..., 1.0

    Usage:
        >>> base_env = make_multi_agent_ac_env(n_agents=2)
        >>> env = DiscretizeActionWrapper(base_env, n_bins=21)
        >>> obs, infos = env.reset()
        >>> actions = {agent: 10 for agent in env.agents}  # 50% of inventory
        >>> obs, rewards, terms, truncs, infos = env.step(actions)
    """

    def __init__(self, env, n_bins: int = 21):
        """
        Initialize wrapper.

        Args:
            env: Base PettingZoo ParallelEnv with continuous action space
            n_bins: Number of discrete action bins (default 21 for 5% increments)
        """
        assert PETTINGZOO_AVAILABLE, "PettingZoo is required for DiscretizeActionWrapper"

        self.env = env
        self.n_bins = n_bins

        # Precompute fraction values for each discrete action
        self.fractions = np.linspace(0, 1, n_bins)

        # Cache discrete action space
        self._discrete_action_space = spaces.Discrete(n_bins)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Space:
        """Return observation space (unchanged from base env)."""
        return self.env.observation_space(agent)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Space:
        """Return discretized action space."""
        return self._discrete_action_space

    @property
    def possible_agents(self):
        return self.env.possible_agents

    @property
    def agents(self):
        return self.env.agents

    @property
    def num_agents(self):
        return len(self.agents)

    @property
    def max_num_agents(self):
        return len(self.possible_agents)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
        """Reset environment."""
        return self.env.reset(seed=seed, options=options)

    def step(
        self,
        actions: Dict[str, int],
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        """
        Step environment with discrete actions.

        Args:
            actions: Dict mapping agent_id -> discrete action index (0 to n_bins-1)

        Returns:
            observations, rewards, terminations, truncations, infos
        """
        # Convert discrete actions to continuous fractions
        continuous_actions = {}
        for agent_id, action in actions.items():
            # Clip action to valid range
            action = int(np.clip(action, 0, self.n_bins - 1))
            fraction = self.fractions[action]
            continuous_actions[agent_id] = np.array([fraction], dtype=np.float32)

        return self.env.step(continuous_actions)

    def render(self):
        """Render environment."""
        return self.env.render()

    def close(self):
        """Close environment."""
        return self.env.close()

    def state(self) -> np.ndarray:
        """Return global state if available."""
        if hasattr(self.env, 'state'):
            return self.env.state()
        else:
            raise NotImplementedError("Base environment does not have state() method")

    def observation_space_global(self) -> spaces.Space:
        """Return global observation space if available."""
        if hasattr(self.env, 'observation_space_global'):
            return self.env.observation_space_global()
        else:
            raise NotImplementedError("Base environment does not have global observation space")

    # Forward other attributes to base environment
    def __getattr__(self, name: str):
        """Forward attribute access to base environment."""
        return getattr(self.env, name)

    @property
    def unwrapped(self):
        """Return unwrapped base environment."""
        return self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env


class SingleAgentDiscretizeWrapper:
    """
    Wrapper that discretizes continuous action space for single-agent Gymnasium envs.

    Converts discrete action indices to continuous fractions in [0, 1].
    """

    def __init__(self, env, n_bins: int = 21):
        """
        Initialize wrapper.

        Args:
            env: Base Gymnasium environment with continuous action space
            n_bins: Number of discrete action bins
        """
        self.env = env
        self.n_bins = n_bins
        self.fractions = np.linspace(0, 1, n_bins)

        # Override action space
        self.action_space = spaces.Discrete(n_bins)

        # Keep observation space from base env
        self.observation_space = env.observation_space

    def reset(self, **kwargs):
        """Reset environment."""
        return self.env.reset(**kwargs)

    def step(self, action: int):
        """Step with discrete action."""
        action = int(np.clip(action, 0, self.n_bins - 1))
        fraction = self.fractions[action]
        continuous_action = np.array([fraction], dtype=np.float32)
        return self.env.step(continuous_action)

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def __getattr__(self, name: str):
        return getattr(self.env, name)

    @property
    def unwrapped(self):
        return self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env


def discretize_action_space(env, n_bins: int = 21):
    """
    Factory function to discretize action space.

    Automatically detects environment type (ParallelEnv vs Gymnasium).

    Args:
        env: Environment to wrap
        n_bins: Number of discrete action bins

    Returns:
        Wrapped environment with discrete action space
    """
    if PETTINGZOO_AVAILABLE and isinstance(env, ParallelEnv):
        return DiscretizeActionWrapper(env, n_bins=n_bins)
    else:
        return SingleAgentDiscretizeWrapper(env, n_bins=n_bins)
