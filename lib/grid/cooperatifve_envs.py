"""
Cooperative multi-agent gridworld environments (PettingZoo AEC).
"""

from __future__ import annotations

from pettingzoo.utils import wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn

from .competitive_envs import CompetitiveGridEnv, EnvConfig


class CooperativeKeyCorridorEnv(CompetitiveGridEnv):
    metadata = {
        "name": "cooperative_keycorridor_v0",
        "render_modes": ["human", "ansi", "rgb_array"],
        "is_parallelizable": True,
    }

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._cooperative = True

    def _build_layout(self, cfg: EnvConfig) -> None:
        super()._build_layout(cfg)
        corridor_row = cfg.height // 2
        self._walls[corridor_row - 1, 1:-1] = True
        self._walls[corridor_row + 1, 1:-1] = True
        self._walls[corridor_row - 1, cfg.width // 3] = False
        self._walls[corridor_row + 1, 2 * cfg.width // 3] = False

        self._door_pos = (corridor_row, cfg.width - 3)
        self._goal_pos = (corridor_row, cfg.width - 2)
        self._key_pos = (corridor_row, 2)
        self._walls[self._door_pos] = False


class CooperativeLavaCrossingEnv(CompetitiveGridEnv):
    metadata = {
        "name": "cooperative_lavacrossing_v0",
        "render_modes": ["human", "ansi", "rgb_array"],
        "is_parallelizable": True,
    }

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._cooperative = True

    def _build_layout(self, cfg: EnvConfig) -> None:
        super()._build_layout(cfg)
        rows = [cfg.height // 3, 2 * cfg.height // 3]
        for r in rows:
            self._lava[r, 1:-1] = True
            gap = self._rng.integers(1, cfg.width - 1)
            self._lava[r, gap] = False

        self._goal_pos = (cfg.height - 2, cfg.width - 2)


def coop_keycorridor_env(**kwargs):
    env_instance = CooperativeKeyCorridorEnv(**kwargs)
    env_instance = wrappers.OrderEnforcingWrapper(env_instance)
    return env_instance


def coop_lavacrossing_env(**kwargs):
    env_instance = CooperativeLavaCrossingEnv(**kwargs)
    env_instance = wrappers.OrderEnforcingWrapper(env_instance)
    return env_instance


parallel_coop_keycorridor_env = parallel_wrapper_fn(coop_keycorridor_env)
parallel_coop_lavacrossing_env = parallel_wrapper_fn(coop_lavacrossing_env)
