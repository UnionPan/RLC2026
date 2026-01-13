"""Planner utilities for cloning simulator state."""

from __future__ import annotations

import copy
from typing import Any, Dict

import numpy as np
from pettingzoo.utils.agent_selector import AgentSelector


def clone_env(env):
    """Best-effort clone for PettingZoo grid envs."""
    try:
        return copy.deepcopy(env)
    except Exception:
        pass

    raw = getattr(env, "unwrapped", env)
    cfg = getattr(raw, "config", None)
    if cfg is None:
        raise

    init_kwargs: Dict[str, Any] = {
        "width": cfg.width,
        "height": cfg.height,
        "view_radius": cfg.view_radius,
        "max_steps": cfg.max_steps,
        "render_mode": None,
    }
    if hasattr(cfg, "wall_density"):
        init_kwargs["wall_density"] = cfg.wall_density
    if hasattr(cfg, "n_pursuers"):
        init_kwargs["n_pursuers"] = cfg.n_pursuers
    if hasattr(cfg, "n_agents"):
        init_kwargs["n_agents"] = cfg.n_agents

    new_env = raw.__class__(**init_kwargs)
    new_env.agents = list(raw.agents)
    new_env.possible_agents = list(raw.possible_agents)
    new_env._steps = raw._steps
    new_env._walls = np.array(raw._walls, copy=True)
    if hasattr(raw, "_lava"):
        new_env._lava = np.array(raw._lava, copy=True)
    new_env._positions = dict(raw._positions)
    new_env._goal_pos = getattr(raw, "_goal_pos", None)
    new_env._key_pos = getattr(raw, "_key_pos", None)
    new_env._door_pos = getattr(raw, "_door_pos", None)
    new_env._door_open = getattr(raw, "_door_open", False)
    new_env._keys_held = dict(getattr(raw, "_keys_held", {}))

    new_env.rewards = dict(raw.rewards)
    new_env.terminations = dict(raw.terminations)
    new_env.truncations = dict(raw.truncations)
    new_env.infos = dict(raw.infos)

    new_env._agent_selector = AgentSelector(new_env.agents)
    new_env.agent_selection = raw.agent_selection
    return new_env
