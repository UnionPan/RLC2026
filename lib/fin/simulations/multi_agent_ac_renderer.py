"""
Pygame Renderer for Multi-Agent Almgren-Chriss Environment (POSG)

Visualizes:
- Shared price evolution with combined impact
- Per-agent inventory trajectories
- Per-agent trading actions (stacked bars)
- Performance comparison (shortfall, rewards)

Author: Yunian Pan
Email: yp1170@nyu.edu
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    pygame = None


# Distinct colors for up to 8 agents
AGENT_COLORS = [
    (100, 200, 255),   # Blue
    (255, 100, 100),   # Red
    (100, 255, 150),   # Green
    (255, 200, 100),   # Orange
    (200, 100, 255),   # Purple
    (255, 255, 100),   # Yellow
    (100, 255, 255),   # Cyan
    (255, 150, 200),   # Pink
]


@dataclass
class MultiAgentRenderConfig:
    """Configuration for multi-agent renderer."""
    width: int = 1400
    height: int = 900
    fps: int = 30

    # Colors (RGB)
    bg_color: Tuple[int, int, int] = (20, 20, 30)
    grid_color: Tuple[int, int, int] = (40, 40, 50)
    text_color: Tuple[int, int, int] = (200, 200, 200)
    price_color: Tuple[int, int, int] = (100, 200, 255)
    impact_color: Tuple[int, int, int] = (255, 80, 80)

    # Agent colors (assigned dynamically)
    agent_colors: List[Tuple[int, int, int]] = field(default_factory=lambda: AGENT_COLORS.copy())

    # Panel layout
    panel_padding: int = 15
    left_panel_width_ratio: float = 0.7  # Main panels
    right_panel_width_ratio: float = 0.3  # Stats panel


class MultiAgentACRenderer:
    """
    Pygame-based renderer for Multi-Agent Almgren-Chriss environment.

    Displays split charts with:
    - Top: shared price index + per-agent trade/slippage markers
    - Bottom: per-agent inventory ratios + cash/wealth indices
    - Footer: leaderboard and live stats
    """

    def __init__(
        self,
        config: Optional[MultiAgentRenderConfig] = None,
        n_agents: int = 2,
        max_steps: int = 20,
        agent_names: Optional[List[str]] = None,
    ):
        if not PYGAME_AVAILABLE:
            raise ImportError(
                "pygame is required for rendering. Install with: pip install pygame"
            )

        self.config = config or MultiAgentRenderConfig()
        self.n_agents = n_agents
        self.max_steps = max_steps
        self.agent_names = agent_names or [f"trader_{i}" for i in range(n_agents)]

        # Assign colors to agents
        self.agent_color_map = {
            name: self.config.agent_colors[i % len(self.config.agent_colors)]
            for i, name in enumerate(self.agent_names)
        }

        # Pygame objects
        self.screen = None
        self.clock = None
        self.font = None
        self.font_small = None
        self.font_large = None
        self.initialized = False

        # Data buffers
        self.reset()

    def reset(self):
        """Reset data buffers for new episode."""
        self.steps: List[int] = []
        self.times: List[float] = []

        # Shared state
        self.prices: List[float] = []  # Unaffected price
        self.impacted_prices: List[float] = []  # Price with impact
        self.aggregate_volumes: List[float] = []

        # Per-agent state
        self.initial_inventories: Dict[str, float] = {n: 0.0 for n in self.agent_names}
        self.initial_notionals: Dict[str, float] = {n: 1.0 for n in self.agent_names}
        self.inventories: Dict[str, List[float]] = {n: [] for n in self.agent_names}
        self.shares_traded: Dict[str, List[float]] = {n: [] for n in self.agent_names}
        self.execution_prices: Dict[str, List[float]] = {n: [] for n in self.agent_names}
        self.slippages_bps: Dict[str, List[float]] = {n: [] for n in self.agent_names}
        self.cash_values: Dict[str, List[float]] = {n: [] for n in self.agent_names}
        self.wealth_values: Dict[str, List[float]] = {n: [] for n in self.agent_names}
        self.rewards: Dict[str, List[float]] = {n: [] for n in self.agent_names}
        self.cumulative_rewards: Dict[str, float] = {n: 0.0 for n in self.agent_names}
        self.shortfalls_bps: Dict[str, List[float]] = {n: [] for n in self.agent_names}

        # Scaling
        self.value_range = (0.8, 1.2)
        self.price_trade_range = (0.9, 1.1)
        self.inv_cash_range = (-0.05, 1.05)
        self.price_range = (95.0, 105.0)
        self.inventory_max = 1_000_000
        self.action_max = 1.0
        self.current_step = 0

    def initialize(self, interactive: bool = True):
        """Initialize pygame display."""
        if self.initialized:
            return

        pygame.init()
        pygame.font.init()

        if interactive:
            self.screen = pygame.display.set_mode(
                (self.config.width, self.config.height)
            )
            pygame.display.set_caption("Multi-Agent Almgren-Chriss Execution")
        else:
            self.screen = pygame.Surface(
                (self.config.width, self.config.height)
            )

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('monospace', 14)
        self.font_small = pygame.font.SysFont('monospace', 11)
        self.font_large = pygame.font.SysFont('monospace', 18, bold=True)
        self.initialized = True

    def update(
        self,
        step: int,
        t: float,
        price: float,
        impacted_price: float,
        agent_data: Dict[str, Dict[str, Any]],
    ):
        """
        Update data buffers with new step data.

        Args:
            step: Current step number
            t: Current time
            price: Unaffected (fundamental) price
            impacted_price: Price after permanent impact
            agent_data: Dict mapping agent name to:
                - 'inventory': remaining inventory
                - 'shares_traded': shares traded this step
                - 'execution_price': execution price this step
                - 'cash': current cash
                - 'reward': reward this step
                - 'shortfall_bps': shortfall in bps
        """
        self.steps.append(step)
        self.times.append(t)
        self.prices.append(price)
        self.impacted_prices.append(impacted_price)

        total_volume = 0
        for agent_name, data in agent_data.items():
            if agent_name in self.agent_names:
                inventory = float(data.get('inventory', 0.0))
                traded = float(data.get('shares_traded', 0.0))
                exec_price = float(data.get('execution_price', impacted_price))
                cash = float(data.get('cash', 0.0))
                reward = float(data.get('reward', 0.0))
                shortfall_bps = float(data.get('shortfall_bps', 0.0))

                if len(self.inventories[agent_name]) == 0:
                    self.initial_inventories[agent_name] = max(inventory, 1e-8)
                    self.initial_notionals[agent_name] = max(inventory * max(price, 1e-8), 1e-8)

                self.inventories[agent_name].append(inventory)
                self.shares_traded[agent_name].append(traded)
                self.execution_prices[agent_name].append(exec_price)
                self.cash_values[agent_name].append(cash)
                self.rewards[agent_name].append(reward)
                self.cumulative_rewards[agent_name] += reward
                self.shortfalls_bps[agent_name].append(shortfall_bps)

                wealth = cash + inventory * impacted_price
                self.wealth_values[agent_name].append(wealth)

                if traded > 0 and impacted_price > 1e-8:
                    slippage_bps = 10000.0 * (impacted_price - exec_price) / impacted_price
                else:
                    slippage_bps = 0.0
                self.slippages_bps[agent_name].append(float(slippage_bps))
                total_volume += traded

        self.aggregate_volumes.append(total_volume)
        self.current_step = step
        self._update_scaling()

    def _update_scaling(self):
        """Update axis scaling based on data."""
        if len(self.prices) == 0:
            return

        price0 = max(float(self.prices[0]), 1e-8)
        price_idx = np.array(self.prices, dtype=float) / price0
        impacted_idx = np.array(self.impacted_prices, dtype=float) / price0
        combined_values = [price_idx, impacted_idx]

        all_invs = []
        all_trades = []
        for agent_name in self.agent_names:
            inv_list = self.inventories[agent_name]
            wealth_list = self.wealth_values[agent_name]
            trade_list = self.shares_traded[agent_name]
            all_invs.extend(inv_list)
            all_trades.extend(trade_list)

            if len(inv_list) > 0:
                inv0 = max(self.initial_inventories[agent_name], 1e-8)
                inv_ratio = np.array(inv_list, dtype=float) / inv0
                combined_values.append(inv_ratio)

            if len(wealth_list) > 0:
                notional0 = max(self.initial_notionals[agent_name], 1e-8)
                wealth_idx = np.array(wealth_list, dtype=float) / notional0
                combined_values.append(wealth_idx)

        flat = np.concatenate(combined_values) if len(combined_values) > 0 else np.array([1.0])
        y_min = float(np.min(flat))
        y_max = float(np.max(flat))
        if abs(y_max - y_min) < 1e-6:
            y_min -= 0.05
            y_max += 0.05
        margin = 0.1 * (y_max - y_min)
        self.value_range = (y_min - margin, y_max + margin)

        # Split-panel scales.
        price_vals = np.concatenate([price_idx, impacted_idx])
        pmin = float(np.min(price_vals))
        pmax = float(np.max(price_vals))
        if abs(pmax - pmin) < 1e-6:
            pmin -= 0.02
            pmax += 0.02
        pmargin = 0.12 * (pmax - pmin)
        self.price_trade_range = (pmin - pmargin, pmax + pmargin)

        inv_cash_series = []
        for agent_name in self.agent_names:
            inv_list = self.inventories[agent_name]
            wealth_list = self.wealth_values[agent_name]
            cash_list = self.cash_values[agent_name]
            if len(inv_list) > 0:
                inv0 = max(self.initial_inventories[agent_name], 1e-8)
                inv_cash_series.append(np.array(inv_list, dtype=float) / inv0)
            if len(wealth_list) > 0:
                notional0 = max(self.initial_notionals[agent_name], 1e-8)
                inv_cash_series.append(np.array(wealth_list, dtype=float) / notional0)
            if len(cash_list) > 0:
                notional0 = max(self.initial_notionals[agent_name], 1e-8)
                inv_cash_series.append(np.array(cash_list, dtype=float) / notional0)

        if len(inv_cash_series) > 0:
            ic = np.concatenate(inv_cash_series)
            icmin = float(np.min(ic))
            icmax = float(np.max(ic))
            if abs(icmax - icmin) < 1e-6:
                icmin -= 0.05
                icmax += 0.05
            icmargin = 0.12 * (icmax - icmin)
            self.inv_cash_range = (icmin - icmargin, icmax + icmargin)

        # Keep legacy scale fields updated.
        all_prices = self.prices + self.impacted_prices
        price_min = min(all_prices)
        price_max = max(all_prices)
        price_margin = (price_max - price_min) * 0.1 + 0.5
        self.price_range = (price_min - price_margin, price_max + price_margin)
        if all_invs:
            self.inventory_max = max(all_invs) * 1.1 + 1000
        if all_trades:
            self.action_max = max(max(all_trades), 1.0)

    def render(self) -> Optional[np.ndarray]:
        """Render current state."""
        if not self.initialized:
            self.initialize()

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return None

        # Clear screen
        self.screen.fill(self.config.bg_color)

        cfg = self.config
        w, h = cfg.width, cfg.height
        pad = cfg.panel_padding
        footer_h = 160
        chart_h = h - 2 * pad - footer_h
        split_gap = 10
        panel_h = (chart_h - split_gap) // 2
        self._draw_price_trade_panel(pad, pad, w - 2 * pad, panel_h)
        self._draw_inventory_cash_panel(pad, pad + panel_h + split_gap, w - 2 * pad, chart_h - panel_h - split_gap)
        self._draw_footer(pad, h - pad - footer_h, w - 2 * pad, footer_h)

        pygame.display.flip()
        self.clock.tick(self.config.fps)

        return pygame.surfarray.array3d(self.screen).transpose(1, 0, 2)

    def _draw_price_trade_panel(self, x: int, y: int, w: int, h: int):
        """Top panel: shared price curves and per-agent trade/slippage markers."""
        cfg = self.config
        ix, iy, iw, ih = self._draw_panel_frame(
            x,
            y,
            w,
            h,
            "Price + Trade/Slippage (normalized)",
        )

        if len(self.prices) == 0:
            self.screen.blit(self.font_small.render("Waiting for data...", True, cfg.text_color), (ix + 8, iy + 8))
            return

        price0 = max(float(self.prices[0]), 1e-8)
        price_idx = np.array(self.prices, dtype=float) / price0
        impacted_idx = np.array(self.impacted_prices, dtype=float) / price0
        y_min, y_max = self.price_trade_range
        if y_max <= y_min:
            y_max = y_min + 1e-3

        def value_to_y(v: float) -> int:
            return int(iy + ih - (v - y_min) / (y_max - y_min) * ih)

        # Grid and labels.
        for i in range(6):
            gy = iy + int(ih * i / 5)
            pygame.draw.line(self.screen, cfg.grid_color, (ix, gy), (ix + iw, gy), 1)
            label_value = y_max - (y_max - y_min) * i / 5
            label = self.font_small.render(f"{label_value:.3f}", True, cfg.text_color)
            self.screen.blit(label, (ix + iw - 55, gy - 8))

        n = len(price_idx)
        if n == 1:
            x_pos = self._step_to_x(0, ix, iw)
            pygame.draw.circle(self.screen, cfg.price_color, (x_pos, value_to_y(float(price_idx[0]))), 3)
            return

        # Shared curves.
        for i in range(1, n):
            x1 = self._step_to_x(i - 1, ix, iw)
            x2 = self._step_to_x(i, ix, iw)
            pygame.draw.line(
                self.screen,
                cfg.price_color,
                (x1, value_to_y(float(price_idx[i - 1]))),
                (x2, value_to_y(float(price_idx[i]))),
                2,
            )
            pygame.draw.line(
                self.screen,
                cfg.impact_color,
                (x1, value_to_y(float(impacted_idx[i - 1]))),
                (x2, value_to_y(float(impacted_idx[i]))),
                2,
            )

        # Trade markers per agent over impacted curve.
        marker_stride = max(1, n // 10)
        for agent_offset, agent_name in enumerate(self.agent_names):
            agent_color = self.agent_color_map[agent_name]
            trades = self.shares_traded[agent_name]
            slippages = self.slippages_bps[agent_name]
            for i in range(1, min(len(trades), n)):
                traded = float(trades[i])
                if traded <= 0:
                    continue

                x_pos = self._step_to_x(i, ix, iw)
                base_y = value_to_y(float(impacted_idx[i]))
                # small deterministic vertical offset to reduce marker overlap
                y_pos = base_y - 3 * (agent_offset % 3)
                vol_ratio = min(1.0, traded / max(self.action_max, 1e-8))
                radius = 3 + int(8 * np.sqrt(max(0.0, vol_ratio)))
                slip = float(slippages[i]) if i < len(slippages) else 0.0
                fill_color = self._slippage_to_color(slip)
                pygame.draw.circle(self.screen, fill_color, (x_pos, y_pos), radius)
                pygame.draw.circle(self.screen, agent_color, (x_pos, y_pos), radius, 1)

                if i % marker_stride == 0 or i == n - 1:
                    lbl = self.font_small.render(f"{agent_name}:{traded/1000:.0f}k|{slip:.1f}bp", True, cfg.text_color)
                    self.screen.blit(lbl, (x_pos + 4, y_pos - 10))

        # Legend.
        lx = ix + 10
        ly = iy + 6
        pygame.draw.line(self.screen, cfg.price_color, (lx, ly + 7), (lx + 14, ly + 7), 2)
        self.screen.blit(self.font_small.render("Fund idx", True, cfg.text_color), (lx + 18, ly))
        lx += 90
        pygame.draw.line(self.screen, cfg.impact_color, (lx, ly + 7), (lx + 14, ly + 7), 2)
        self.screen.blit(self.font_small.render("Imp idx", True, cfg.text_color), (lx + 18, ly))
        lx += 82
        self.screen.blit(
            self.font_small.render("Markers: size=vol, color=slip, ring=agent", True, cfg.text_color),
            (lx, ly),
        )

    def _draw_inventory_cash_panel(self, x: int, y: int, w: int, h: int):
        """Bottom panel: per-agent inventory, cash, and wealth indices."""
        cfg = self.config
        ix, iy, iw, ih = self._draw_panel_frame(
            x,
            y,
            w,
            h,
            "Inventory + Cash/Wealth (normalized)",
        )

        if len(self.prices) == 0:
            self.screen.blit(self.font_small.render("Waiting for data...", True, cfg.text_color), (ix + 8, iy + 8))
            return

        y_min, y_max = self.inv_cash_range
        if y_max <= y_min:
            y_max = y_min + 1e-3

        def value_to_y(v: float) -> int:
            return int(iy + ih - (v - y_min) / (y_max - y_min) * ih)

        # Grid and labels.
        for i in range(6):
            gy = iy + int(ih * i / 5)
            pygame.draw.line(self.screen, cfg.grid_color, (ix, gy), (ix + iw, gy), 1)
            label_value = y_max - (y_max - y_min) * i / 5
            label = self.font_small.render(f"{label_value:.3f}", True, cfg.text_color)
            self.screen.blit(label, (ix + iw - 55, gy - 8))

        # Per-agent curves.
        for agent_name in self.agent_names:
            color = self.agent_color_map[agent_name]
            wealth_color = self._lighten_color(color, 0.45)
            cash_color = self._lighten_color(color, 0.25)
            inv_list = self.inventories[agent_name]
            cash_list = self.cash_values[agent_name]
            wealth_list = self.wealth_values[agent_name]
            if len(inv_list) < 2:
                continue

            inv0 = max(self.initial_inventories[agent_name], 1e-8)
            notional0 = max(self.initial_notionals[agent_name], 1e-8)
            inv_ratio = np.array(inv_list, dtype=float) / inv0
            cash_idx = np.array(cash_list, dtype=float) / notional0
            wealth_idx = np.array(wealth_list, dtype=float) / notional0

            n = len(inv_ratio)
            for i in range(1, n):
                x1 = self._step_to_x(i - 1, ix, iw)
                x2 = self._step_to_x(i, ix, iw)
                pygame.draw.line(
                    self.screen,
                    color,
                    (x1, value_to_y(float(inv_ratio[i - 1]))),
                    (x2, value_to_y(float(inv_ratio[i]))),
                    1,
                )
                pygame.draw.line(
                    self.screen,
                    cash_color,
                    (x1, value_to_y(float(cash_idx[i - 1]))),
                    (x2, value_to_y(float(cash_idx[i]))),
                    1,
                )
                pygame.draw.line(
                    self.screen,
                    wealth_color,
                    (x1, value_to_y(float(wealth_idx[i - 1]))),
                    (x2, value_to_y(float(wealth_idx[i]))),
                    2,
                )

        # Agent chips.
        chip_x = ix + 8
        chip_y = iy + 6
        for agent_name in self.agent_names:
            color = self.agent_color_map[agent_name]
            pygame.draw.rect(self.screen, color, (chip_x, chip_y + 3, 10, 10))
            self.screen.blit(self.font_small.render(agent_name, True, cfg.text_color), (chip_x + 14, chip_y))
            chip_x += 96
            if chip_x > ix + iw - 90:
                chip_x = ix + 8
                chip_y += 14

    def _draw_unified_panel(self, x: int, y: int, w: int, h: int):
        """Draw unified normalized chart for price, inventory, wealth, and markers."""
        cfg = self.config
        ix, iy, iw, ih = self._draw_panel_frame(
            x,
            y,
            w,
            h,
            "Unified Multi-Agent View (Price / Inventory / Wealth + Trade Markers)",
        )

        if len(self.prices) == 0:
            self.screen.blit(self.font_small.render("Waiting for data...", True, cfg.text_color), (ix + 8, iy + 8))
            return

        price0 = max(float(self.prices[0]), 1e-8)
        price_idx = np.array(self.prices, dtype=float) / price0
        impacted_idx = np.array(self.impacted_prices, dtype=float) / price0
        y_min, y_max = self.value_range
        if y_max <= y_min:
            y_max = y_min + 1e-3

        def value_to_y(v: float) -> int:
            return int(iy + ih - (v - y_min) / (y_max - y_min) * ih)

        # Grid and y labels.
        for i in range(6):
            gy = iy + int(ih * i / 5)
            pygame.draw.line(self.screen, cfg.grid_color, (ix, gy), (ix + iw, gy), 1)
            label_value = y_max - (y_max - y_min) * i / 5
            label = self.font_small.render(f"{label_value:.3f}", True, cfg.text_color)
            self.screen.blit(label, (ix + iw - 55, gy - 8))

        n = len(self.prices)

        # Shared price curves.
        for i in range(1, n):
            x1 = self._step_to_x(i - 1, ix, iw)
            x2 = self._step_to_x(i, ix, iw)
            pygame.draw.line(
                self.screen,
                cfg.price_color,
                (x1, value_to_y(float(price_idx[i - 1]))),
                (x2, value_to_y(float(price_idx[i]))),
                2,
            )
            pygame.draw.line(
                self.screen,
                cfg.impact_color,
                (x1, value_to_y(float(impacted_idx[i - 1]))),
                (x2, value_to_y(float(impacted_idx[i]))),
                2,
            )

        # Per-agent inventory + wealth curves.
        for agent_name in self.agent_names:
            color = self.agent_color_map[agent_name]
            wealth_color = self._lighten_color(color, 0.45)
            inv_list = self.inventories[agent_name]
            wealth_list = self.wealth_values[agent_name]
            if len(inv_list) < 2:
                continue

            inv0 = max(self.initial_inventories[agent_name], 1e-8)
            notional0 = max(self.initial_notionals[agent_name], 1e-8)
            inv_ratio = np.array(inv_list, dtype=float) / inv0
            wealth_idx = np.array(wealth_list, dtype=float) / notional0

            for i in range(1, len(inv_list)):
                x1 = self._step_to_x(i - 1, ix, iw)
                x2 = self._step_to_x(i, ix, iw)
                pygame.draw.line(
                    self.screen,
                    color,
                    (x1, value_to_y(float(inv_ratio[i - 1]))),
                    (x2, value_to_y(float(inv_ratio[i]))),
                    1,
                )
                pygame.draw.line(
                    self.screen,
                    wealth_color,
                    (x1, value_to_y(float(wealth_idx[i - 1]))),
                    (x2, value_to_y(float(wealth_idx[i]))),
                    2,
                )

        # Trade markers: radius=volume, color=slippage, ring=agent color.
        marker_stride = max(1, n // 10)
        for agent_name in self.agent_names:
            agent_color = self.agent_color_map[agent_name]
            trades = self.shares_traded[agent_name]
            slippages = self.slippages_bps[agent_name]
            wealth_list = self.wealth_values[agent_name]
            if len(wealth_list) == 0:
                continue
            notional0 = max(self.initial_notionals[agent_name], 1e-8)
            wealth_idx = np.array(wealth_list, dtype=float) / notional0

            for i in range(1, len(wealth_idx)):
                traded = float(trades[i]) if i < len(trades) else 0.0
                if traded <= 0:
                    continue
                x_pos = self._step_to_x(i, ix, iw)
                y_pos = value_to_y(float(wealth_idx[i]))
                vol_ratio = min(1.0, traded / max(self.action_max, 1e-8))
                radius = 3 + int(8 * np.sqrt(max(0.0, vol_ratio)))
                slip = float(slippages[i]) if i < len(slippages) else 0.0
                fill_color = self._slippage_to_color(slip)
                pygame.draw.circle(self.screen, fill_color, (x_pos, y_pos), radius)
                pygame.draw.circle(self.screen, agent_color, (x_pos, y_pos), radius, 1)

                if i % marker_stride == 0 or i == len(wealth_idx) - 1:
                    lbl = self.font_small.render(
                        f"{agent_name}:{traded/1000:.0f}k|{slip:.1f}bp",
                        True,
                        cfg.text_color,
                    )
                    self.screen.blit(lbl, (x_pos + 4, y_pos - 10))

        # Compact legend.
        lx = ix + 10
        ly = iy + 6
        pygame.draw.line(self.screen, cfg.price_color, (lx, ly + 7), (lx + 14, ly + 7), 2)
        self.screen.blit(self.font_small.render("Fund", True, cfg.text_color), (lx + 18, ly))
        lx += 75
        pygame.draw.line(self.screen, cfg.impact_color, (lx, ly + 7), (lx + 14, ly + 7), 2)
        self.screen.blit(self.font_small.render("Imp", True, cfg.text_color), (lx + 18, ly))
        lx += 65
        self.screen.blit(
            self.font_small.render("Inv=agent color, Wealth=light agent color, Marker=size vol / color slip", True, cfg.text_color),
            (lx, ly),
        )

        # Agent chips for color mapping.
        chip_x = ix + 10
        chip_y = iy + 24
        for agent_name in self.agent_names:
            color = self.agent_color_map[agent_name]
            pygame.draw.rect(self.screen, color, (chip_x, chip_y + 3, 10, 10))
            self.screen.blit(self.font_small.render(agent_name, True, cfg.text_color), (chip_x + 14, chip_y))
            chip_x += 105
            if chip_x > ix + iw - 120:
                chip_x = ix + 10
                chip_y += 14

    def _draw_footer(self, x: int, y: int, w: int, h: int):
        """Draw leaderboard and per-agent summary."""
        cfg = self.config
        rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(self.screen, cfg.grid_color, rect, 1)

        if len(self.prices) == 0:
            return

        current_time = self.times[-1] if len(self.times) > 0 else 0.0
        total_volume = self.aggregate_volumes[-1] if len(self.aggregate_volumes) > 0 else 0.0
        top_line = (
            f"Step {self.current_step}/{self.max_steps}   "
            f"t={current_time:.3f}   "
            f"Fund ${self.prices[-1]:.2f}   "
            f"Imp ${self.impacted_prices[-1]:.2f}   "
            f"Market volume {total_volume:,.0f}"
        )
        self.screen.blit(self.font.render(top_line, True, cfg.text_color), (x + 10, y + 8))

        sorted_agents = sorted(
            self.agent_names,
            key=lambda a: self.cumulative_rewards.get(a, 0.0),
            reverse=True,
        )

        row_y = y + 34
        for rank, agent_name in enumerate(sorted_agents):
            color = self.agent_color_map[agent_name]
            inv = self.inventories[agent_name][-1] if self.inventories[agent_name] else 0.0
            inv0 = max(self.initial_inventories[agent_name], 1e-8)
            inv_pct = 100.0 * inv / inv0
            cash = self.cash_values[agent_name][-1] if self.cash_values[agent_name] else 0.0
            wealth = self.wealth_values[agent_name][-1] if self.wealth_values[agent_name] else 0.0
            notional0 = max(self.initial_notionals[agent_name], 1e-8)
            wealth_idx = wealth / notional0
            traded = self.shares_traded[agent_name][-1] if self.shares_traded[agent_name] else 0.0
            slip = self.slippages_bps[agent_name][-1] if self.slippages_bps[agent_name] else 0.0
            shortfall = self.shortfalls_bps[agent_name][-1] if self.shortfalls_bps[agent_name] else 0.0
            reward = self.cumulative_rewards[agent_name]

            line = (
                f"#{rank+1} {agent_name}  "
                f"Inv {inv_pct:6.2f}%  "
                f"Cash ${cash:,.0f}  "
                f"Wealth {wealth_idx:.3f}x  "
                f"Trade {traded:,.0f}  "
                f"Slip {slip:+.2f}bp  "
                f"SF {shortfall:+.2f}bp  "
                f"Reward {reward:+.4f}"
            )
            pygame.draw.rect(self.screen, color, (x + 10, row_y + 4, 9, 9))
            self.screen.blit(self.font_small.render(line, True, cfg.text_color), (x + 24, row_y))
            row_y += 18
            if row_y > y + h - 20:
                break

    def _lighten_color(self, color: Tuple[int, int, int], factor: float = 0.4) -> Tuple[int, int, int]:
        """Return a lighter shade of an RGB color."""
        r, g, b = color
        return (
            int(r + (255 - r) * factor),
            int(g + (255 - g) * factor),
            int(b + (255 - b) * factor),
        )

    def _slippage_to_color(self, slippage_bps: float) -> Tuple[int, int, int]:
        """Map slippage to marker color (green=better, red=worse)."""
        magnitude = min(abs(slippage_bps) / 40.0, 1.0)
        if slippage_bps >= 0:
            return (255, int(200 - 120 * magnitude), int(120 - 70 * magnitude))
        return (int(120 - 60 * magnitude), int(220 - 20 * magnitude), int(120 - 40 * magnitude))

    def _draw_panel_frame(self, x: int, y: int, w: int, h: int, title: str):
        """Draw panel background and title."""
        cfg = self.config

        rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(self.screen, cfg.grid_color, rect, 1)

        title_surf = self.font.render(title, True, cfg.text_color)
        self.screen.blit(title_surf, (x + 5, y + 2))

        return (x + 5, y + 22, w - 10, h - 27)

    def _step_to_x(self, step: int, x_start: int, w: int) -> int:
        """Convert step index to x coordinate."""
        n = len(self.steps)
        if n <= 1:
            return x_start
        return int(x_start + (step / max(1, self.max_steps)) * w * 0.9)

    def _draw_price_panel(self, x: int, y: int, w: int, h: int):
        """Draw price panel with unaffected and impacted prices."""
        cfg = self.config
        ix, iy, iw, ih = self._draw_panel_frame(x, y, w, h, "Price Evolution (Shared)")

        if len(self.prices) < 2:
            return

        price_min, price_max = self.price_range

        def price_to_y(p: float) -> int:
            return int(iy + ih - (p - price_min) / (price_max - price_min) * ih)

        # Grid lines
        for i in range(5):
            gy = iy + int(ih * i / 4)
            pygame.draw.line(self.screen, cfg.grid_color, (ix, gy), (ix + iw, gy), 1)
            price_label = price_max - (price_max - price_min) * i / 4
            label = self.font_small.render(f"${price_label:.2f}", True, cfg.text_color)
            self.screen.blit(label, (ix + iw - 55, gy - 6))

        # Draw lines
        for i in range(1, len(self.prices)):
            x1 = self._step_to_x(i - 1, ix, iw)
            x2 = self._step_to_x(i, ix, iw)

            # Unaffected price
            y1 = price_to_y(self.prices[i - 1])
            y2 = price_to_y(self.prices[i])
            pygame.draw.line(self.screen, cfg.price_color, (x1, y1), (x2, y2), 2)

            # Impacted price
            iy1 = price_to_y(self.impacted_prices[i - 1])
            iy2 = price_to_y(self.impacted_prices[i])
            pygame.draw.line(self.screen, cfg.impact_color, (x1, iy1), (x2, iy2), 2)

        # Legend
        pygame.draw.line(self.screen, cfg.price_color, (ix + 10, iy + 5), (ix + 30, iy + 5), 2)
        self.screen.blit(self.font_small.render("Fundamental", True, cfg.text_color), (ix + 35, iy))
        pygame.draw.line(self.screen, cfg.impact_color, (ix + 120, iy + 5), (ix + 140, iy + 5), 2)
        self.screen.blit(self.font_small.render("With Impact", True, cfg.text_color), (ix + 145, iy))

        # Current values
        if len(self.prices) > 0:
            spread = self.prices[-1] - self.impacted_prices[-1]
            label = f"Impact: ${spread:.3f}"
            self.screen.blit(self.font_small.render(label, True, cfg.text_color), (ix + iw - 150, iy))

    def _draw_inventory_panel(self, x: int, y: int, w: int, h: int):
        """Draw per-agent inventory trajectories."""
        cfg = self.config
        ix, iy, iw, ih = self._draw_panel_frame(x, y, w, h, "Inventory Trajectories")

        if len(self.steps) < 2:
            return

        inv_max = self.inventory_max

        def inv_to_y(inv: float) -> int:
            return int(iy + ih - (inv / inv_max) * ih * 0.95)

        # Grid
        for i in range(5):
            gy = iy + int(ih * i / 4)
            pygame.draw.line(self.screen, cfg.grid_color, (ix, gy), (ix + iw, gy), 1)

        # Draw each agent's trajectory
        legend_x = ix + 10
        for agent_name in self.agent_names:
            color = self.agent_color_map[agent_name]
            inv_list = self.inventories[agent_name]

            if len(inv_list) < 2:
                continue

            for i in range(1, len(inv_list)):
                x1 = self._step_to_x(i - 1, ix, iw)
                x2 = self._step_to_x(i, ix, iw)
                y1 = inv_to_y(inv_list[i - 1])
                y2 = inv_to_y(inv_list[i])
                pygame.draw.line(self.screen, color, (x1, y1), (x2, y2), 2)

            # Legend entry
            pygame.draw.line(self.screen, color, (legend_x, iy + 5), (legend_x + 20, iy + 5), 2)
            self.screen.blit(self.font_small.render(agent_name, True, cfg.text_color), (legend_x + 25, iy))
            legend_x += 100

    def _draw_action_panel(self, x: int, y: int, w: int, h: int):
        """Draw stacked bar chart of trading actions."""
        cfg = self.config
        ix, iy, iw, ih = self._draw_panel_frame(x, y, w, h, "Trading Actions (per step)")

        if len(self.steps) < 2:
            return

        action_max = self.action_max
        n_steps = len(self.steps)
        bar_width = max(4, int(iw * 0.8 / max(1, self.max_steps)))

        # Draw stacked bars
        for step_idx in range(1, n_steps):
            x_pos = self._step_to_x(step_idx, ix, iw)
            y_bottom = iy + ih

            # Stack bars for each agent
            for agent_name in self.agent_names:
                color = self.agent_color_map[agent_name]
                traded = self.shares_traded[agent_name][step_idx] if step_idx < len(self.shares_traded[agent_name]) else 0

                bar_h = int((traded / action_max) * ih * 0.85)
                if bar_h > 0:
                    bar_rect = pygame.Rect(x_pos - bar_width//2, y_bottom - bar_h, bar_width, bar_h)
                    pygame.draw.rect(self.screen, color, bar_rect)
                    y_bottom -= bar_h

        # Total volume label
        if len(self.aggregate_volumes) > 0:
            total = self.aggregate_volumes[-1]
            label = f"Total volume: {total:,.0f}"
            self.screen.blit(self.font_small.render(label, True, cfg.text_color), (ix + iw - 180, iy))

    def _draw_stats_panel(self, x: int, y: int, w: int, h: int):
        """Draw agent statistics and leaderboard."""
        cfg = self.config
        ix, iy, iw, ih = self._draw_panel_frame(x, y, w, h, "Agent Performance")

        # Step counter
        step_text = f"Step: {self.current_step}/{self.max_steps}"
        self.screen.blit(self.font_large.render(step_text, True, cfg.text_color), (ix + 5, iy + 5))

        # Time
        if len(self.times) > 0:
            time_text = f"Time: {self.times[-1]:.3f}"
            self.screen.blit(self.font.render(time_text, True, cfg.text_color), (ix + 5, iy + 30))

        # Separator
        pygame.draw.line(self.screen, cfg.grid_color, (ix, iy + 55), (ix + iw - 10, iy + 55), 1)

        # Agent stats
        y_offset = iy + 65
        row_height = 85

        # Sort agents by cumulative reward (leaderboard)
        sorted_agents = sorted(
            self.agent_names,
            key=lambda a: self.cumulative_rewards.get(a, 0),
            reverse=True
        )

        for rank, agent_name in enumerate(sorted_agents):
            color = self.agent_color_map[agent_name]

            # Rank and name
            rank_text = f"#{rank + 1}"
            pygame.draw.rect(self.screen, color, (ix + 5, y_offset, 8, 8))
            self.screen.blit(self.font.render(f"{rank_text} {agent_name}", True, color), (ix + 18, y_offset - 3))

            # Stats
            stats_y = y_offset + 18

            # Inventory
            inv = self.inventories[agent_name][-1] if self.inventories[agent_name] else 0
            inv_text = f"Inventory: {inv:,.0f}"
            self.screen.blit(self.font_small.render(inv_text, True, cfg.text_color), (ix + 10, stats_y))

            # Cash
            cash = self.cash_values[agent_name][-1] if self.cash_values[agent_name] else 0
            cash_text = f"Cash: ${cash:,.0f}"
            self.screen.blit(self.font_small.render(cash_text, True, cfg.text_color), (ix + 10, stats_y + 14))

            # Cumulative reward
            cum_reward = self.cumulative_rewards[agent_name]
            reward_text = f"Total Reward: {cum_reward:.4f}"
            self.screen.blit(self.font_small.render(reward_text, True, cfg.text_color), (ix + 10, stats_y + 28))

            # Shortfall
            shortfall = self.shortfalls_bps[agent_name][-1] if self.shortfalls_bps[agent_name] else 0
            shortfall_text = f"Shortfall: {shortfall:.1f} bps"
            shortfall_color = (255, 100, 100) if shortfall > 50 else (100, 255, 100)
            self.screen.blit(self.font_small.render(shortfall_text, True, shortfall_color), (ix + 10, stats_y + 42))

            # Separator
            y_offset += row_height
            if rank < len(sorted_agents) - 1:
                pygame.draw.line(self.screen, cfg.grid_color, (ix + 5, y_offset - 5), (ix + iw - 15, y_offset - 5), 1)

        # Competition summary at bottom
        summary_y = iy + ih - 60
        pygame.draw.line(self.screen, cfg.grid_color, (ix, summary_y - 10), (ix + iw - 10, summary_y - 10), 1)

        if len(sorted_agents) >= 2:
            leader = sorted_agents[0]
            lead_amount = self.cumulative_rewards[leader] - self.cumulative_rewards[sorted_agents[1]]
            lead_text = f"Leader: {leader}"
            self.screen.blit(self.font.render(lead_text, True, self.agent_color_map[leader]), (ix + 5, summary_y))
            margin_text = f"Lead: {lead_amount:+.4f}"
            self.screen.blit(self.font_small.render(margin_text, True, cfg.text_color), (ix + 5, summary_y + 20))

    def close(self):
        """Clean up pygame resources."""
        if self.initialized:
            pygame.quit()
            self.screen = None
            self.clock = None
            self.font = None
            self.font_small = None
            self.font_large = None
            self.initialized = False

    def wait_until_closed(self, idle_fps: int = 30):
        """
        Keep the last rendered frame visible until user closes the window.

        Args:
            idle_fps: Event-loop tick rate while waiting.
        """
        if not self.initialized or self.screen is None:
            return

        while self.initialized:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.close()
                    return

            pygame.display.flip()
            if self.clock is not None:
                self.clock.tick(max(1, idle_fps))
