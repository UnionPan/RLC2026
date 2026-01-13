"""Dataset utilities for grid autoencoder training."""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from lib.grid import (
    competitive_fourrooms_env,
    competitive_obstructedmaze_env,
    coop_keycorridor_env,
    coop_lavacrossing_env,
    pursuit_env,
)


class GridRolloutDataset(Dataset):
    """Dataset of grid observations stored as an array."""

    def __init__(self, observations: np.ndarray):
        self.observations = observations.astype(np.float32)

    def __len__(self) -> int:
        return self.observations.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.observations[idx])


def collect_rollouts(
    env_name: str,
    episodes: int = 100,
    max_steps: int = 200,
    seed: int = 0,
    width: int = 11,
    height: int = 11,
    view_radius: int = 2,
    wall_density: float = 0.1,
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """Collect observations from grid environments."""
    env = _make_env(
        env_name,
        width=width,
        height=height,
        view_radius=view_radius,
        max_steps=max_steps,
        wall_density=wall_density,
    )

    observations: List[np.ndarray] = []
    env.reset(seed=seed)
    for ep in range(episodes):
        if ep > 0:
            env.reset(seed=seed + ep)
        for agent_id in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            if obs is not None:
                observations.append(obs)
            action = None if (termination or truncation) else env.action_space(agent_id).sample()
            env.step(action)

    env.close()

    data = np.stack(observations, axis=0)
    obs_shape = data.shape[1:]
    return data, obs_shape


def save_rollouts(path: str, observations: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, observations)


def load_rollouts(path: str) -> np.ndarray:
    return np.load(path)


def _make_env(
    env_name: str,
    width: int,
    height: int,
    view_radius: int,
    max_steps: int,
    wall_density: float,
):
    if env_name == "competitive_fourrooms":
        return competitive_fourrooms_env(
            width=width,
            height=height,
            view_radius=view_radius,
            max_steps=max_steps,
        )
    if env_name == "competitive_obstructedmaze":
        return competitive_obstructedmaze_env(
            width=width,
            height=height,
            view_radius=view_radius,
            max_steps=max_steps,
            wall_density=wall_density,
        )
    if env_name == "coop_keycorridor":
        return coop_keycorridor_env(
            width=width,
            height=height,
            view_radius=view_radius,
            max_steps=max_steps,
        )
    if env_name == "coop_lavacrossing":
        return coop_lavacrossing_env(
            width=width,
            height=height,
            view_radius=view_radius,
            max_steps=max_steps,
        )
    if env_name == "pursuit_evasion":
        return pursuit_env(
            width=width,
            height=height,
            n_pursuers=1,
            view_radius=view_radius,
            max_steps=max_steps,
            wall_density=wall_density,
        )
    raise ValueError(f"Unknown grid env: {env_name}")
