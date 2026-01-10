"""
Partially observable pursuit-evasion gridworld (PettingZoo AEC).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn


ACTION_TO_DELTA = {
    0: (0, 0),   # stay
    1: (-1, 0),  # up
    2: (1, 0),   # down
    3: (0, -1),  # left
    4: (0, 1),   # right
}


@dataclass
class EnvConfig:
    width: int = 11
    height: int = 11
    n_pursuers: int = 1
    view_radius: int = 2
    max_steps: int = 200
    wall_density: float = 0.1
    capture_reward: float = 1.0
    evader_survival_reward: float = 0.0
    evader_escape_reward: float = 0.5


class raw_env(AECEnv):
    metadata = {
        "name": "pursuit_evasion_v0",
        "render_modes": ["human", "ansi", "rgb_array"],
        "is_parallelizable": True,
    }

    def __init__(
        self,
        width: int = 11,
        height: int = 11,
        n_pursuers: int = 1,
        view_radius: int = 2,
        max_steps: int = 200,
        wall_density: float = 0.1,
        capture_reward: float = 1.0,
        evader_survival_reward: float = 0.0,
        evader_escape_reward: float = 0.5,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.config = EnvConfig(
            width=width,
            height=height,
            n_pursuers=n_pursuers,
            view_radius=view_radius,
            max_steps=max_steps,
            wall_density=wall_density,
            capture_reward=capture_reward,
            evader_survival_reward=evader_survival_reward,
            evader_escape_reward=evader_escape_reward,
        )
        self.render_mode = render_mode

        self.possible_agents = [f"pursuer_{i}" for i in range(n_pursuers)] + ["evader_0"]
        self.agents: List[str] = []

        view_size = 2 * view_radius + 1
        self._obs_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(view_size, view_size, 4),
            dtype=np.float32,
        )
        self._act_space = spaces.Discrete(5)

        self._rng = np.random.default_rng()
        self._walls = np.zeros((height, width), dtype=bool)
        self._positions: Dict[str, Tuple[int, int]] = {}
        self._agent_selector = None
        self._steps = 0
        self._pygame = None
        self._screen = None
        self._clock = None
        self._cell_size = 32

        self.rewards: Dict[str, float] = {}
        self.terminations: Dict[str, bool] = {}
        self.truncations: Dict[str, bool] = {}
        self.infos: Dict[str, Dict] = {}

    def observation_space(self, agent: str):
        return self._obs_space

    def action_space(self, agent: str):
        return self._act_space

    def seed(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.seed(seed)

        cfg = self.config
        self.agents = list(self.possible_agents)
        self._steps = 0

        self._build_walls(cfg)
        self._place_agents(cfg)

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

    def _build_walls(self, cfg: EnvConfig) -> None:
        self._walls = np.zeros((cfg.height, cfg.width), dtype=bool)
        self._walls[0, :] = True
        self._walls[-1, :] = True
        self._walls[:, 0] = True
        self._walls[:, -1] = True

        interior = [
            (r, c)
            for r in range(1, cfg.height - 1)
            for c in range(1, cfg.width - 1)
        ]
        n_walls = int(len(interior) * cfg.wall_density)
        if n_walls > 0:
            chosen = self._rng.choice(len(interior), size=n_walls, replace=False)
            for idx in chosen:
                r, c = interior[idx]
                self._walls[r, c] = True

    def _place_agents(self, cfg: EnvConfig) -> None:
        free_cells = [
            (r, c)
            for r in range(1, cfg.height - 1)
            for c in range(1, cfg.width - 1)
            if not self._walls[r, c]
        ]
        if len(free_cells) < len(self.possible_agents):
            raise ValueError("Not enough free cells to place agents")

        chosen = self._rng.choice(len(free_cells), size=len(self.possible_agents), replace=False)
        self._positions = {}
        for agent, idx in zip(self.possible_agents, chosen):
            self._positions[agent] = free_cells[idx]

    def observe(self, agent: str):
        cfg = self.config
        view_size = 2 * cfg.view_radius + 1
        obs = np.zeros((view_size, view_size, 4), dtype=np.float32)

        center_r, center_c = self._positions[agent]
        for dr in range(-cfg.view_radius, cfg.view_radius + 1):
            for dc in range(-cfg.view_radius, cfg.view_radius + 1):
                rr = center_r + dr
                cc = center_c + dc
                vr = dr + cfg.view_radius
                vc = dc + cfg.view_radius

                if 0 <= rr < cfg.height and 0 <= cc < cfg.width:
                    obs[vr, vc, 3] = 1.0
                    if self._walls[rr, cc]:
                        obs[vr, vc, 0] = 1.0
                    for pursuer in self._pursuers():
                        if self._positions[pursuer] == (rr, cc):
                            obs[vr, vc, 1] = 1.0
                    if self._positions["evader_0"] == (rr, cc):
                        obs[vr, vc, 2] = 1.0
                else:
                    obs[vr, vc, 0] = 1.0
        return obs

    def _pursuers(self) -> List[str]:
        return [agent for agent in self.agents if agent.startswith("pursuer_")]

    def _move(self, agent: str, action: int) -> None:
        dr, dc = ACTION_TO_DELTA.get(int(action), (0, 0))
        r, c = self._positions[agent]
        nr, nc = r + dr, c + dc
        if self._walls[nr, nc]:
            return
        self._positions[agent] = (nr, nc)

    def step(self, action):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        self._move(agent, action)

        self.rewards = {a: 0.0 for a in self.agents}

        if self._is_captured():
            for pursuer in self._pursuers():
                self.rewards[pursuer] = self.config.capture_reward
            self.rewards["evader_0"] = -self.config.capture_reward
            self.terminations = {a: True for a in self.agents}

        if not any(self.terminations.values()):
            self.rewards["evader_0"] += self.config.evader_survival_reward

        if self._agent_selector.is_last():
            self._steps += 1
            if self._steps >= self.config.max_steps and not any(self.terminations.values()):
                self.truncations = {a: True for a in self.agents}
                self.rewards["evader_0"] += self.config.evader_escape_reward

        self.agent_selection = self._agent_selector.next()

    def _is_captured(self) -> bool:
        evader_pos = self._positions["evader_0"]
        return any(self._positions[pursuer] == evader_pos for pursuer in self._pursuers())

    def render(self):
        if self.render_mode not in {"human", "ansi", None}:
            if self.render_mode != "rgb_array":
                raise ValueError(f"Unsupported render_mode: {self.render_mode}")

        grid = np.full((self.config.height, self.config.width), " ", dtype="<U1")
        grid[self._walls] = "#"
        for pursuer in self._pursuers():
            r, c = self._positions[pursuer]
            grid[r, c] = "P"
        r, c = self._positions["evader_0"]
        grid[r, c] = "E"

        lines = ["".join(row) for row in grid]
        output = "\n".join(lines)

        if self.render_mode == "ansi" or self.render_mode is None:
            return output
        if self.render_mode == "human":
            return self._render_pygame()
        if self.render_mode == "rgb_array":
            return self._render_pygame(rgb_array=True)
        return output

    def _render_pygame(self, rgb_array: bool = False):
        if self._pygame is None:
            try:
                import pygame  # pylint: disable=import-error
            except ImportError as exc:
                raise ImportError("pygame is required for render_mode='human' or 'rgb_array'") from exc
            self._pygame = pygame
            if self.render_mode == "human":
                pygame.display.init()
            pygame.font.init()
            self._clock = pygame.time.Clock()

        height_px = self.config.height * self._cell_size
        width_px = self.config.width * self._cell_size

        if self._screen is None:
            if self.render_mode == "human":
                self._screen = self._pygame.display.set_mode((width_px, height_px))
                self._pygame.display.set_caption("Pursuit-Evasion")
            else:
                self._screen = self._pygame.Surface((width_px, height_px))

        self._screen.fill((20, 20, 20))

        wall_color = (40, 40, 40)
        floor_color = (230, 230, 230)
        pursuer_color = (30, 90, 200)
        evader_color = (200, 60, 60)
        grid_color = (180, 180, 180)

        for r in range(self.config.height):
            for c in range(self.config.width):
                rect = self._pygame.Rect(
                    c * self._cell_size,
                    r * self._cell_size,
                    self._cell_size,
                    self._cell_size,
                )
                if self._walls[r, c]:
                    self._pygame.draw.rect(self._screen, wall_color, rect)
                else:
                    self._pygame.draw.rect(self._screen, floor_color, rect)
                self._pygame.draw.rect(self._screen, grid_color, rect, 1)

        for pursuer in self._pursuers():
            r, c = self._positions[pursuer]
            center = (
                c * self._cell_size + self._cell_size // 2,
                r * self._cell_size + self._cell_size // 2,
            )
            radius = self._cell_size // 3
            self._pygame.draw.circle(self._screen, pursuer_color, center, radius)

        r, c = self._positions["evader_0"]
        rect = self._pygame.Rect(
            c * self._cell_size + self._cell_size // 4,
            r * self._cell_size + self._cell_size // 4,
            self._cell_size // 2,
            self._cell_size // 2,
        )
        self._pygame.draw.rect(self._screen, evader_color, rect)

        if self.render_mode == "human":
            self._pygame.display.flip()
            self._clock.tick(30)

        if rgb_array:
            frame = self._pygame.surfarray.array3d(self._screen)
            return np.transpose(frame, (1, 0, 2))
        return None

    def close(self):
        if self._pygame is not None:
            if self._screen is not None:
                self._screen = None
            if self._clock is not None:
                self._clock = None
            self._pygame.quit()
            self._pygame = None

    def render_ansi(self):
        self.render_mode = "ansi"
        return self.render()

    def state(self):
        grid = np.zeros((self.config.height, self.config.width, 3), dtype=np.float32)
        grid[:, :, 0] = self._walls.astype(np.float32)
        for pursuer in self._pursuers():
            r, c = self._positions[pursuer]
            grid[r, c, 1] = 1.0
        r, c = self._positions["evader_0"]
        grid[r, c, 2] = 1.0
        return grid


def env(**kwargs):
    env_instance = raw_env(**kwargs)
    env_instance = wrappers.OrderEnforcingWrapper(env_instance)
    return env_instance


parallel_env = parallel_wrapper_fn(env)
