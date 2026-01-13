"""
Competitive multi-agent gridworld environments (PettingZoo AEC).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import AgentSelector
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
    n_agents: int = 2
    view_radius: int = 2
    max_steps: int = 200
    wall_density: float = 0.1
    render_mode: Optional[str] = None


class CompetitiveGridEnv(AECEnv):
    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "is_parallelizable": True,
    }

    def __init__(
        self,
        width: int = 11,
        height: int = 11,
        n_agents: int = 2,
        view_radius: int = 2,
        max_steps: int = 200,
        wall_density: float = 0.1,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.config = EnvConfig(
            width=width,
            height=height,
            n_agents=n_agents,
            view_radius=view_radius,
            max_steps=max_steps,
            wall_density=wall_density,
            render_mode=render_mode,
        )
        self.render_mode = render_mode

        self.possible_agents = [f"agent_{i}" for i in range(n_agents)]
        self.agents: List[str] = []

        view_size = 2 * view_radius + 1
        self._obs_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(view_size, view_size, 9),
            dtype=np.float32,
        )
        self._act_space = spaces.Discrete(5)

        self._rng = np.random.default_rng()
        self._walls = np.zeros((height, width), dtype=bool)
        self._lava = np.zeros((height, width), dtype=bool)
        self._positions: Dict[str, Tuple[int, int]] = {}
        self._goal_pos: Optional[Tuple[int, int]] = None
        self._key_pos: Optional[Tuple[int, int]] = None
        self._door_pos: Optional[Tuple[int, int]] = None
        self._door_open = False
        self._keys_held: Dict[str, bool] = {}
        self._agent_selector = None
        self._steps = 0
        self._pygame = None
        self._screen = None
        self._clock = None
        self._font = None
        self._cell_size = 32
        self._history: List[Dict[str, Tuple[int, int]]] = []
        self._replay_active = False
        self._replay_index = 0
        self._hold_on_finish = True
        self._hold_active = False
        self._cooperative = False

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
        self._door_open = False
        self._keys_held = {agent: False for agent in self.agents}

        self._build_layout(cfg)
        self._place_agents(cfg)

        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.next()
        self._history = [self._snapshot_positions()]
        self._replay_active = False
        self._replay_index = 0
        self._hold_active = False

        self.rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self._clear_rewards()

    def _build_layout(self, cfg: EnvConfig) -> None:
        self._walls = np.zeros((cfg.height, cfg.width), dtype=bool)
        self._lava = np.zeros((cfg.height, cfg.width), dtype=bool)
        self._goal_pos = None
        self._key_pos = None
        self._door_pos = None
        self._walls[0, :] = True
        self._walls[-1, :] = True
        self._walls[:, 0] = True
        self._walls[:, -1] = True

    def _place_agents(self, cfg: EnvConfig) -> None:
        blocked = {pos for pos in (self._goal_pos, self._key_pos, self._door_pos) if pos is not None}
        free_cells = [
            (r, c)
            for r in range(1, cfg.height - 1)
            for c in range(1, cfg.width - 1)
            if not self._walls[r, c] and not self._lava[r, c] and (r, c) not in blocked
        ]
        needed = len(self.possible_agents) + (1 if self._goal_pos is None else 0)
        if len(free_cells) < needed:
            raise ValueError("Not enough free cells to place agents and goal")

        chosen = self._rng.choice(len(free_cells), size=needed, replace=False)
        self._positions = {}
        for agent, idx in zip(self.possible_agents, chosen[: len(self.possible_agents)]):
            self._positions[agent] = free_cells[idx]

        if self._goal_pos is None:
            goal_idx = chosen[-1]
            self._goal_pos = free_cells[goal_idx]

    def observe(self, agent: str):
        cfg = self.config
        view_size = 2 * cfg.view_radius + 1
        obs = np.zeros((view_size, view_size, 9), dtype=np.float32)

        center_r, center_c = self._positions[agent]
        for dr in range(-cfg.view_radius, cfg.view_radius + 1):
            for dc in range(-cfg.view_radius, cfg.view_radius + 1):
                rr = center_r + dr
                cc = center_c + dc
                vr = dr + cfg.view_radius
                vc = dc + cfg.view_radius

                if 0 <= rr < cfg.height and 0 <= cc < cfg.width:
                    if self._is_visible((center_r, center_c), (rr, cc)):
                        obs[vr, vc, 7] = 1.0
                        if self._walls[rr, cc]:
                            obs[vr, vc, 0] = 1.0
                        if self._lava[rr, cc]:
                            obs[vr, vc, 1] = 1.0
                        if self._goal_pos == (rr, cc):
                            obs[vr, vc, 2] = 1.0
                        if self._key_pos == (rr, cc):
                            obs[vr, vc, 3] = 1.0
                        if self._door_pos == (rr, cc) and not self._door_open:
                            obs[vr, vc, 4] = 1.0
                        if (rr, cc) == (center_r, center_c):
                            obs[vr, vc, 5] = 1.0
                        else:
                            for other in self.agents:
                                if other != agent and self._positions[other] == (rr, cc):
                                    obs[vr, vc, 6] = 1.0
                else:
                    obs[vr, vc, 0] = 1.0
        if self._keys_held.get(agent, False):
            obs[:, :, 8] = 1.0
        return obs

    def _line_cells(self, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        r0, c0 = start
        r1, c1 = end
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        step_r = 1 if r1 >= r0 else -1
        step_c = 1 if c1 >= c0 else -1
        err = dr - dc
        r, c = r0, c0
        cells = []
        if dr == 0 and dc == 0:
            return cells
        while (r, c) != (r1, c1):
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r += step_r
                cells.append((r, c))
            if e2 < dr:
                err += dr
                c += step_c
                if not cells or cells[-1] != (r, c):
                    cells.append((r, c))
        return cells

    def _is_visible(self, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        line = self._line_cells(start, end)
        if not line:
            return True
        for r, c in line[:-1]:
            if self._walls[r, c]:
                return False
        return True

    def _snapshot_positions(self) -> Dict[str, Tuple[int, int]]:
        return dict(self._positions)

    def _is_blocked(self, agent: str, nr: int, nc: int) -> bool:
        if self._walls[nr, nc]:
            return True
        if self._door_pos == (nr, nc) and not self._door_open:
            if self._keys_held.get(agent, False):
                self._door_open = True
                return False
            return True
        return False

    def _move(self, agent: str, action: int) -> None:
        dr, dc = ACTION_TO_DELTA.get(int(action), (0, 0))
        r, c = self._positions[agent]
        nr, nc = r + dr, c + dc
        if self._is_blocked(agent, nr, nc):
            return
        self._positions[agent] = (nr, nc)
        if self._key_pos == (nr, nc):
            self._keys_held[agent] = True
            self._key_pos = None

    def _is_goal(self, agent: str) -> bool:
        return self._goal_pos is not None and self._positions[agent] == self._goal_pos

    def _is_lava(self, agent: str) -> bool:
        r, c = self._positions[agent]
        return self._lava[r, c]

    def _set_winner(self, winner: str, reward: float = 1.0) -> None:
        self.rewards = {a: -reward for a in self.agents}
        self.rewards[winner] = reward
        self.terminations = {a: True for a in self.agents}

    def _set_shared_outcome(self, reward: float) -> None:
        self.rewards = {a: reward for a in self.agents}
        self.terminations = {a: True for a in self.agents}

    def _all_at_goal(self) -> bool:
        if self._goal_pos is None:
            return False
        return all(self._positions[a] == self._goal_pos for a in self.agents)

    def step(self, action):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        self._move(agent, action)

        self.rewards = {a: 0.0 for a in self.agents}

        if self._cooperative:
            if any(self._is_lava(a) for a in self.agents):
                self._set_shared_outcome(reward=-1.0)
            elif self._all_at_goal():
                self._set_shared_outcome(reward=1.0)
        else:
            if self._is_lava(agent):
                winner = next(a for a in self.agents if a != agent)
                self._set_winner(winner, reward=1.0)
            elif self._is_goal(agent):
                self._set_winner(agent, reward=1.0)

        if self._agent_selector.is_last():
            self._steps += 1
            if self._steps >= self.config.max_steps and not any(self.terminations.values()):
                self.truncations = {a: True for a in self.agents}

        self._accumulate_rewards()
        self.agent_selection = self._agent_selector.next()
        self._history.append(self._snapshot_positions())

    def render(self):
        if self.render_mode not in {"human", "ansi", None}:
            if self.render_mode != "rgb_array":
                raise ValueError(f"Unsupported render_mode: {self.render_mode}")

        grid = np.full((self.config.height, self.config.width), " ", dtype="<U1")
        grid[self._walls] = "#"
        grid[self._lava] = "L"
        if self._goal_pos is not None:
            r, c = self._goal_pos
            grid[r, c] = "G"
        if self._key_pos is not None:
            r, c = self._key_pos
            grid[r, c] = "K"
        if self._door_pos is not None and not self._door_open:
            r, c = self._door_pos
            grid[r, c] = "D"

        if self._replay_active and self._history:
            positions = self._history[self._replay_index]
        else:
            positions = self._positions

        for agent, (r, c) in positions.items():
            label = "A" if agent == "agent_0" else "B"
            grid[r, c] = label

        lines = ["".join(row) for row in grid]
        output = "\n".join(lines)

        if self.render_mode == "ansi" or self.render_mode is None:
            return output
        if self.render_mode == "human":
            if (
                self._hold_on_finish
                and not self._hold_active
                and (any(self.terminations.values()) or any(self.truncations.values()))
            ):
                self._hold_active = True
            return self._render_pygame()
        if self.render_mode == "rgb_array":
            return self._render_pygame(rgb_array=True)
        return output

    def _handle_pygame_events(self, button_rect) -> bool:
        for event in self._pygame.event.get():
            if event.type == self._pygame.QUIT:
                self.close()
                return False
            if event.type == self._pygame.MOUSEBUTTONDOWN and event.button == 1:
                if button_rect.collidepoint(event.pos):
                    if self._replay_active:
                        self._replay_active = False
                        self._replay_index = max(0, len(self._history) - 1)
                    else:
                        self._replay_active = True
                        self._replay_index = 0
        return True

    def _draw_pygame(self, positions, button_rect) -> None:
        self._screen.fill((20, 20, 20))

        wall_color = (40, 40, 40)
        floor_color = (230, 230, 230)
        lava_color = (200, 80, 40)
        goal_color = (60, 160, 60)
        key_color = (220, 180, 40)
        door_color = (110, 80, 40)
        agent0_color = (30, 90, 200)
        agent1_color = (200, 60, 60)
        grid_color = (180, 180, 180)
        overlay_colors = {
            "agent_0": (40, 120, 240, 90),
            "agent_1": (240, 80, 80, 90),
        }

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
                elif self._lava[r, c]:
                    self._pygame.draw.rect(self._screen, lava_color, rect)
                else:
                    self._pygame.draw.rect(self._screen, floor_color, rect)
                self._pygame.draw.rect(self._screen, grid_color, rect, 1)

        overlay = self._pygame.Surface(self._screen.get_size(), flags=self._pygame.SRCALPHA)
        radius = self.config.view_radius
        for agent, (r, c) in positions.items():
            color = overlay_colors.get(agent, (120, 120, 120, 90))
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < self.config.height and 0 <= cc < self.config.width:
                        if self._is_visible((r, c), (rr, cc)):
                            rect = self._pygame.Rect(
                                cc * self._cell_size,
                                rr * self._cell_size,
                                self._cell_size,
                                self._cell_size,
                            )
                            self._pygame.draw.rect(overlay, color, rect)
        self._screen.blit(overlay, (0, 0))

        if self._goal_pos is not None:
            r, c = self._goal_pos
            rect = self._pygame.Rect(
                c * self._cell_size + self._cell_size // 4,
                r * self._cell_size + self._cell_size // 4,
                self._cell_size // 2,
                self._cell_size // 2,
            )
            self._pygame.draw.rect(self._screen, goal_color, rect)

        if self._key_pos is not None:
            r, c = self._key_pos
            center = (
                c * self._cell_size + self._cell_size // 2,
                r * self._cell_size + self._cell_size // 2,
            )
            radius = self._cell_size // 4
            self._pygame.draw.circle(self._screen, key_color, center, radius)

        if self._door_pos is not None and not self._door_open:
            r, c = self._door_pos
            rect = self._pygame.Rect(
                c * self._cell_size + self._cell_size // 8,
                r * self._cell_size + self._cell_size // 8,
                self._cell_size * 3 // 4,
                self._cell_size * 3 // 4,
            )
            self._pygame.draw.rect(self._screen, door_color, rect)

        for agent, (r, c) in positions.items():
            color = agent0_color if agent == "agent_0" else agent1_color
            center = (
                c * self._cell_size + self._cell_size // 2,
                r * self._cell_size + self._cell_size // 2,
            )
            radius = self._cell_size // 3
            self._pygame.draw.circle(self._screen, color, center, radius)

        button_color = (50, 50, 50)
        button_border = (220, 220, 220)
        self._pygame.draw.rect(self._screen, button_color, button_rect)
        self._pygame.draw.rect(self._screen, button_border, button_rect, 1)
        label = "Replay" if not self._replay_active else "Stop"
        text_surf = self._font.render(label, True, (240, 240, 240))
        text_rect = text_surf.get_rect(center=button_rect.center)
        self._screen.blit(text_surf, text_rect)

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
            self._font = pygame.font.SysFont(None, 20)

        height_px = self.config.height * self._cell_size
        width_px = self.config.width * self._cell_size

        if self._screen is None:
            if self.render_mode == "human":
                self._screen = self._pygame.display.set_mode((width_px, height_px))
                self._pygame.display.set_caption("Competitive Gridworld")
            else:
                self._screen = self._pygame.Surface((width_px, height_px))

        button_margin = 6
        button_w = 90
        button_h = 28
        button_rect = self._pygame.Rect(
            button_margin,
            button_margin,
            button_w,
            button_h,
        )

        if self.render_mode == "human":
            if not self._handle_pygame_events(button_rect):
                return None

        if self._replay_active and self._history:
            positions = self._history[self._replay_index]
        else:
            positions = self._positions

        self._draw_pygame(positions, button_rect)

        if self.render_mode == "human":
            self._pygame.display.flip()
            self._clock.tick(30)

        if self._replay_active and self._history:
            self._replay_index += 1
            if self._replay_index >= len(self._history):
                self._replay_active = False
                self._replay_index = max(0, len(self._history) - 1)

        if self._hold_active and self.render_mode == "human":
            while True:
                if not self._handle_pygame_events(button_rect):
                    return None
                if self._replay_active and self._history:
                    positions = self._history[self._replay_index]
                else:
                    positions = self._positions
                self._draw_pygame(positions, button_rect)
                self._pygame.display.flip()
                self._clock.tick(30)
                if self._replay_active and self._history:
                    self._replay_index += 1
                    if self._replay_index >= len(self._history):
                        self._replay_active = False
                        self._replay_index = max(0, len(self._history) - 1)

        if rgb_array:
            frame = self._pygame.surfarray.array3d(self._screen)
            return np.transpose(frame, (1, 0, 2))
        return None

    def plot_trajectories(self, path: Optional[str] = None) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is required for trajectory plots.")
            return

        plt.figure(figsize=(6, 6))
        plt.imshow(self._walls, cmap="gray_r", origin="upper")

        lava_cells = np.argwhere(self._lava)
        if lava_cells.size:
            plt.scatter(lava_cells[:, 1], lava_cells[:, 0], c="orangered", s=10, label="lava")

        if self._goal_pos is not None:
            plt.scatter(self._goal_pos[1], self._goal_pos[0], c="green", s=60, marker="s", label="goal")
        if self._key_pos is not None:
            plt.scatter(self._key_pos[1], self._key_pos[0], c="gold", s=50, marker="o", label="key")
        if self._door_pos is not None and not self._door_open:
            plt.scatter(self._door_pos[1], self._door_pos[0], c="saddlebrown", s=60, marker="s", label="door")

        for agent in self.possible_agents:
            positions = [snap[agent] for snap in self._history if agent in snap]
            if positions:
                rows = [p[0] for p in positions]
                cols = [p[1] for p in positions]
                plt.plot(cols, rows, marker="o", markersize=3, linewidth=1, label=agent)

        plt.gca().invert_yaxis()
        plt.legend(loc="upper right")
        plt.title("Trajectories")
        plt.tight_layout()
        if path:
            plt.savefig(path, dpi=150)
        else:
            plt.show()

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


class CompetitiveFourRoomsEnv(CompetitiveGridEnv):
    metadata = {
        "name": "competitive_fourrooms_v0",
        "render_modes": ["human", "ansi", "rgb_array"],
        "is_parallelizable": True,
    }

    def _build_layout(self, cfg: EnvConfig) -> None:
        super()._build_layout(cfg)
        mid_r = cfg.height // 2
        mid_c = cfg.width // 2
        self._walls[mid_r, :] = True
        self._walls[:, mid_c] = True
        openings = [
            (mid_r, cfg.width // 4),
            (mid_r, 3 * cfg.width // 4),
            (cfg.height // 4, mid_c),
            (3 * cfg.height // 4, mid_c),
        ]
        for r, c in openings:
            self._walls[r, c] = False

        free_cells = [
            (r, c)
            for r in range(1, cfg.height - 1)
            for c in range(1, cfg.width - 1)
            if not self._walls[r, c]
        ]
        self._goal_pos = free_cells[self._rng.integers(len(free_cells))]


class CompetitiveObstructedMazeEnv(CompetitiveGridEnv):
    metadata = {
        "name": "competitive_obstructedmaze_v0",
        "render_modes": ["human", "ansi", "rgb_array"],
        "is_parallelizable": True,
    }

    def _build_layout(self, cfg: EnvConfig) -> None:
        super()._build_layout(cfg)
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

        free_cells = [
            (r, c)
            for r in range(1, cfg.height - 1)
            for c in range(1, cfg.width - 1)
            if not self._walls[r, c]
        ]
        self._goal_pos = free_cells[self._rng.integers(len(free_cells))]


def competitive_fourrooms_env(**kwargs):
    env_instance = CompetitiveFourRoomsEnv(**kwargs)
    env_instance = wrappers.OrderEnforcingWrapper(env_instance)
    return env_instance


def competitive_obstructedmaze_env(**kwargs):
    env_instance = CompetitiveObstructedMazeEnv(**kwargs)
    env_instance = wrappers.OrderEnforcingWrapper(env_instance)
    return env_instance


parallel_competitive_fourrooms_env = parallel_wrapper_fn(competitive_fourrooms_env)
parallel_competitive_obstructedmaze_env = parallel_wrapper_fn(competitive_obstructedmaze_env)
