"""
Pygame Renderer for Almgren-Chriss Optimal Execution Environment

Visualizes:
- Price evolution (unaffected vs execution price with impact)
- Inventory trajectory (actual vs optimal)
- Trading actions and slippage
- Cash accumulation and implementation shortfall

Author: Yunian Pan
Email: yp1170@nyu.edu
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    pygame = None


@dataclass
class RenderConfig:
    """Configuration for the renderer."""
    width: int = 1200
    height: int = 800
    fps: int = 30

    # Colors (RGB)
    bg_color: Tuple[int, int, int] = (20, 20, 30)
    grid_color: Tuple[int, int, int] = (40, 40, 50)
    text_color: Tuple[int, int, int] = (200, 200, 200)

    # Price colors
    price_color: Tuple[int, int, int] = (100, 200, 255)  # Blue - unaffected
    exec_price_color: Tuple[int, int, int] = (255, 100, 100)  # Red - execution

    # Inventory colors
    inventory_color: Tuple[int, int, int] = (100, 255, 150)  # Green - actual
    optimal_color: Tuple[int, int, int] = (255, 200, 100)  # Orange - optimal

    # Action colors
    action_color: Tuple[int, int, int] = (200, 100, 255)  # Purple
    slippage_color: Tuple[int, int, int] = (255, 50, 50)  # Red

    # Cash colors
    cash_color: Tuple[int, int, int] = (100, 255, 200)  # Cyan

    # Panel layout
    panel_padding: int = 20
    panel_height_ratio: float = 0.25  # Each panel is 25% of height


class AlmgrenChrissRenderer:
    """
    Pygame-based renderer for Almgren-Chriss environment.

    Displays a unified chart with:
    - Price index (unaffected + execution)
    - Inventory ratio
    - Cash index + wealth index
    - Trade markers (radius=volume, color=slippage)
    """

    def __init__(
        self,
        config: Optional[RenderConfig] = None,
        max_steps: int = 20,
    ):
        """
        Initialize renderer.

        Args:
            config: Render configuration
            max_steps: Maximum steps in episode (for x-axis scaling)
        """
        if not PYGAME_AVAILABLE:
            raise ImportError(
                "pygame is required for rendering. Install with: pip install pygame"
            )

        self.config = config or RenderConfig()
        self.max_steps = max_steps

        # Pygame objects
        self.screen = None
        self.clock = None
        self.font = None
        self.font_small = None
        self.initialized = False

        # Data buffers
        self.reset()

    def reset(self):
        """Reset data buffers for new episode."""
        self.times: List[float] = []
        self.prices: List[float] = []
        self.exec_prices: List[float] = []
        self.inventories: List[float] = []
        self.optimal_inventories: List[float] = []
        self.shares_traded: List[float] = []
        self.cash_values: List[float] = []
        self.wealth_values: List[float] = []
        self.shortfalls: List[float] = []
        self.slippages: List[float] = []  # Per-trade slippage

        # Scaling parameters (updated dynamically)
        self.value_range = (0.8, 1.2)
        self.price_trade_range = (0.9, 1.1)
        self.inv_cash_range = (-0.05, 1.05)
        # Legacy fields kept for compatibility with older panel helpers.
        self.price_range = (90.0, 110.0)
        self.inventory_max = 1_000_000
        self.cash_max = 100_000_000
        self.action_max = 1.0

    def initialize(self, interactive: bool = True):
        """
        Initialize pygame display.

        Args:
            interactive: If True, create display window
        """
        if self.initialized:
            return

        pygame.init()
        pygame.font.init()

        if interactive:
            self.screen = pygame.display.set_mode(
                (self.config.width, self.config.height)
            )
            pygame.display.set_caption("Almgren-Chriss Optimal Execution")
        else:
            self.screen = pygame.Surface(
                (self.config.width, self.config.height)
            )

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('monospace', 16)
        self.font_small = pygame.font.SysFont('monospace', 12)
        self.initialized = True

    def update(
        self,
        t: float,
        price: float,
        exec_price: float,
        inventory: float,
        optimal_inventory: float,
        shares_traded: float,
        cash: float,
        shortfall: float,
        arrival_price: float,
    ):
        """
        Update data buffers with new step data.

        Args:
            t: Current time
            price: Unaffected price
            exec_price: Execution price (with impact)
            inventory: Remaining inventory
            optimal_inventory: Optimal inventory at this time
            shares_traded: Shares traded this step
            cash: Current cash
            shortfall: Implementation shortfall
            arrival_price: Arrival price for slippage calc
        """
        self.times.append(t)
        self.prices.append(price)
        self.exec_prices.append(exec_price)
        self.inventories.append(inventory)
        self.optimal_inventories.append(optimal_inventory)
        self.shares_traded.append(shares_traded)
        self.cash_values.append(cash)
        wealth = cash + inventory * price
        self.wealth_values.append(wealth)
        self.shortfalls.append(shortfall)

        # Per-trade slippage (bps)
        if shares_traded > 0:
            slippage_per_share = arrival_price - exec_price
            slippage_bps = 10000 * slippage_per_share / arrival_price
        else:
            slippage_bps = 0
        self.slippages.append(slippage_bps)

        # Update scaling
        self._update_scaling()

    def _update_scaling(self):
        """Update axis scaling based on data."""
        if len(self.prices) == 0 or len(self.inventories) == 0:
            return

        price0 = max(self.prices[0], 1e-8)
        inv0 = max(self.inventories[0], 1e-8)
        notional0 = max(price0 * inv0, 1e-8)

        price_idx = np.array(self.prices, dtype=float) / price0
        exec_idx = np.array(self.exec_prices, dtype=float) / price0
        inv_ratio = np.array(self.inventories, dtype=float) / inv0
        cash_idx = np.array(self.cash_values, dtype=float) / notional0
        wealth_idx = np.array(self.wealth_values, dtype=float) / notional0

        combined = np.concatenate([price_idx, exec_idx, inv_ratio, cash_idx, wealth_idx])
        y_min = float(np.min(combined))
        y_max = float(np.max(combined))
        if abs(y_max - y_min) < 1e-6:
            y_min -= 0.05
            y_max += 0.05
        margin = 0.1 * (y_max - y_min)
        self.value_range = (y_min - margin, y_max + margin)

        # Split-panel scales.
        price_vals = np.concatenate([price_idx, exec_idx])
        pmin = float(np.min(price_vals))
        pmax = float(np.max(price_vals))
        if abs(pmax - pmin) < 1e-6:
            pmin -= 0.02
            pmax += 0.02
        pmargin = 0.12 * (pmax - pmin)
        self.price_trade_range = (pmin - pmargin, pmax + pmargin)

        inv_cash_vals = np.concatenate([inv_ratio, cash_idx, wealth_idx])
        icmin = float(np.min(inv_cash_vals))
        icmax = float(np.max(inv_cash_vals))
        if abs(icmax - icmin) < 1e-6:
            icmin -= 0.05
            icmax += 0.05
        icmargin = 0.12 * (icmax - icmin)
        self.inv_cash_range = (icmin - icmargin, icmax + icmargin)

        # Keep raw-scale ranges updated for legacy panel methods.
        raw_price_min = float(min(np.min(self.prices), np.min(self.exec_prices)))
        raw_price_max = float(max(np.max(self.prices), np.max(self.exec_prices)))
        raw_price_margin = 0.1 * (raw_price_max - raw_price_min) + 1.0
        self.price_range = (raw_price_min - raw_price_margin, raw_price_max + raw_price_margin)
        self.inventory_max = max(float(np.max(self.inventories)) * 1.1, 1.0)
        self.cash_max = max(float(np.max(np.abs(self.cash_values))) * 1.2 + 1000.0, 1.0)

        if len(self.shares_traded) > 0:
            self.action_max = max(float(np.max(self.shares_traded)), 1.0)

    def render(self) -> Optional[np.ndarray]:
        """
        Render current state.

        Returns:
            RGB array if not interactive, None otherwise
        """
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
        footer_h = 80
        chart_h = h - 2 * pad - footer_h
        split_gap = 10
        panel_h = (chart_h - split_gap) // 2
        self._draw_price_trade_panel(pad, pad, w - 2 * pad, panel_h)
        self._draw_inventory_cash_panel(pad, pad + panel_h + split_gap, w - 2 * pad, chart_h - panel_h - split_gap)
        self._draw_footer(pad, h - pad - footer_h, w - 2 * pad, footer_h)

        pygame.display.flip()
        self.clock.tick(self.config.fps)

        # Return RGB array
        return pygame.surfarray.array3d(self.screen).transpose(1, 0, 2)

    def _draw_price_trade_panel(self, x: int, y: int, w: int, h: int):
        """Top panel: price curves with trade/slippage markers."""
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

        price0 = max(self.prices[0], 1e-8)
        price_idx = np.array(self.prices, dtype=float) / price0
        exec_idx = np.array(self.exec_prices, dtype=float) / price0
        y_min, y_max = self.price_trade_range
        if y_max <= y_min:
            y_max = y_min + 1e-3

        def value_to_y(v: float) -> int:
            return int(iy + ih - (v - y_min) / (y_max - y_min) * ih)

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
                cfg.exec_price_color,
                (x1, value_to_y(float(exec_idx[i - 1]))),
                (x2, value_to_y(float(exec_idx[i]))),
                2,
            )

        # Mark trades on execution curve.
        marker_stride = max(1, n // 12)
        for i in range(1, n):
            traded = float(self.shares_traded[i]) if i < len(self.shares_traded) else 0.0
            if traded <= 0:
                continue

            x_pos = self._step_to_x(i, ix, iw)
            y_pos = value_to_y(float(exec_idx[i]))
            vol_ratio = min(1.0, traded / max(self.action_max, 1e-8))
            radius = 3 + int(8 * np.sqrt(max(0.0, vol_ratio)))
            slip = float(self.slippages[i]) if i < len(self.slippages) else 0.0
            color = self._slippage_to_color(slip)
            pygame.draw.circle(self.screen, color, (x_pos, y_pos), radius)
            pygame.draw.circle(self.screen, cfg.bg_color, (x_pos, y_pos), max(1, radius - 2), 1)

            if i % marker_stride == 0 or i == n - 1:
                marker_txt = self.font_small.render(f"{traded/1000:.0f}k|{slip:.1f}bp", True, cfg.text_color)
                self.screen.blit(marker_txt, (x_pos + 5, y_pos - 10))

        # Legend
        lx = ix + 8
        ly = iy + 6
        pygame.draw.line(self.screen, cfg.price_color, (lx, ly + 7), (lx + 14, ly + 7), 2)
        self.screen.blit(self.font_small.render("Price idx", True, cfg.text_color), (lx + 18, ly))
        lx += 110
        pygame.draw.line(self.screen, cfg.exec_price_color, (lx, ly + 7), (lx + 14, ly + 7), 2)
        self.screen.blit(self.font_small.render("Exec idx", True, cfg.text_color), (lx + 18, ly))
        lx += 100
        pygame.draw.circle(self.screen, (255, 130, 80), (lx + 7, ly + 7), 4)
        self.screen.blit(
            self.font_small.render("Trade marker: size=volume, color=slippage", True, cfg.text_color),
            (lx + 18, ly),
        )

    def _draw_inventory_cash_panel(self, x: int, y: int, w: int, h: int):
        """Bottom panel: inventory, cash index, and wealth index."""
        cfg = self.config
        ix, iy, iw, ih = self._draw_panel_frame(
            x,
            y,
            w,
            h,
            "Inventory + Cash/Wealth (normalized)",
        )

        if len(self.inventories) == 0:
            self.screen.blit(self.font_small.render("Waiting for data...", True, cfg.text_color), (ix + 8, iy + 8))
            return

        price0 = max(self.prices[0], 1e-8)
        inv0 = max(self.inventories[0], 1e-8)
        notional0 = max(price0 * inv0, 1e-8)

        inv_ratio = np.array(self.inventories, dtype=float) / inv0
        cash_idx = np.array(self.cash_values, dtype=float) / notional0
        wealth_idx = np.array(self.wealth_values, dtype=float) / notional0
        y_min, y_max = self.inv_cash_range
        if y_max <= y_min:
            y_max = y_min + 1e-3

        def value_to_y(v: float) -> int:
            return int(iy + ih - (v - y_min) / (y_max - y_min) * ih)

        for i in range(6):
            gy = iy + int(ih * i / 5)
            pygame.draw.line(self.screen, cfg.grid_color, (ix, gy), (ix + iw, gy), 1)
            label_value = y_max - (y_max - y_min) * i / 5
            label = self.font_small.render(f"{label_value:.3f}", True, cfg.text_color)
            self.screen.blit(label, (ix + iw - 55, gy - 8))

        n = len(inv_ratio)
        if n == 1:
            x_pos = self._step_to_x(0, ix, iw)
            pygame.draw.circle(self.screen, cfg.inventory_color, (x_pos, value_to_y(float(inv_ratio[0]))), 3)
            return

        series = [
            (inv_ratio, cfg.inventory_color, 2, "Inventory"),
            (cash_idx, cfg.cash_color, 2, "Cash idx"),
            (wealth_idx, cfg.optimal_color, 3, "Wealth idx"),
        ]
        for values, color, width, _ in series:
            for i in range(1, n):
                x1 = self._step_to_x(i - 1, ix, iw)
                x2 = self._step_to_x(i, ix, iw)
                pygame.draw.line(
                    self.screen,
                    color,
                    (x1, value_to_y(float(values[i - 1]))),
                    (x2, value_to_y(float(values[i]))),
                    width,
                )

        lx = ix + 8
        ly = iy + 6
        legend = [
            ("Inventory", cfg.inventory_color),
            ("Cash idx", cfg.cash_color),
            ("Wealth idx", cfg.optimal_color),
        ]
        for label, color in legend:
            pygame.draw.line(self.screen, color, (lx, ly + 7), (lx + 14, ly + 7), 2)
            self.screen.blit(self.font_small.render(label, True, cfg.text_color), (lx + 18, ly))
            lx += 112

    def _draw_unified_panel(self, x: int, y: int, w: int, h: int):
        """Draw normalized price, inventory, wealth with trade/slippage overlay."""
        cfg = self.config
        ix, iy, iw, ih = self._draw_panel_frame(
            x,
            y,
            w,
            h,
            "Unified Execution View (Price / Inventory / Wealth + Trade Markers)",
        )

        if len(self.prices) == 0:
            self.screen.blit(self.font_small.render("Waiting for data...", True, cfg.text_color), (ix + 8, iy + 8))
            return

        price0 = max(self.prices[0], 1e-8)
        inv0 = max(self.inventories[0], 1e-8)
        notional0 = max(price0 * inv0, 1e-8)

        price_idx = np.array(self.prices, dtype=float) / price0
        exec_idx = np.array(self.exec_prices, dtype=float) / price0
        inv_ratio = np.array(self.inventories, dtype=float) / inv0
        cash_idx = np.array(self.cash_values, dtype=float) / notional0
        wealth_idx = np.array(self.wealth_values, dtype=float) / notional0

        y_min, y_max = self.value_range
        if y_max <= y_min:
            y_max = y_min + 1e-3

        def value_to_y(v: float) -> int:
            return int(iy + ih - (v - y_min) / (y_max - y_min) * ih)

        # Horizontal grid and labels.
        for i in range(6):
            gy = iy + int(ih * i / 5)
            pygame.draw.line(self.screen, cfg.grid_color, (ix, gy), (ix + iw, gy), 1)
            label_value = y_max - (y_max - y_min) * i / 5
            label = self.font_small.render(f"{label_value:.3f}", True, cfg.text_color)
            self.screen.blit(label, (ix + iw - 55, gy - 8))

        n = len(self.prices)
        if n == 1:
            x_pos = self._step_to_x(0, ix, iw)
            y_pos = value_to_y(float(wealth_idx[0]))
            pygame.draw.circle(self.screen, cfg.cash_color, (x_pos, y_pos), 4)
            return

        # Core curves.
        series = [
            (price_idx, cfg.price_color, 2, "Price idx"),
            (exec_idx, cfg.exec_price_color, 1, "Exec idx"),
            (inv_ratio, cfg.inventory_color, 2, "Inventory"),
            (cash_idx, cfg.cash_color, 2, "Cash idx"),
            (wealth_idx, cfg.optimal_color, 3, "Wealth idx"),
        ]
        for values, color, width, _ in series:
            for i in range(1, n):
                x1 = self._step_to_x(i - 1, ix, iw)
                x2 = self._step_to_x(i, ix, iw)
                y1 = value_to_y(float(values[i - 1]))
                y2 = value_to_y(float(values[i]))
                pygame.draw.line(self.screen, color, (x1, y1), (x2, y2), width)

        # Trade markers on wealth curve.
        # Radius ~ traded volume; color ~ slippage bps.
        marker_stride = max(1, n // 12)
        for i in range(1, n):
            traded = float(self.shares_traded[i]) if i < len(self.shares_traded) else 0.0
            if traded <= 0:
                continue

            x_pos = self._step_to_x(i, ix, iw)
            y_pos = value_to_y(float(wealth_idx[i]))
            vol_ratio = min(1.0, traded / max(self.action_max, 1e-8))
            radius = 3 + int(9 * np.sqrt(max(0.0, vol_ratio)))
            slip = float(self.slippages[i]) if i < len(self.slippages) else 0.0
            color = self._slippage_to_color(slip)
            pygame.draw.circle(self.screen, color, (x_pos, y_pos), radius)
            pygame.draw.circle(self.screen, cfg.bg_color, (x_pos, y_pos), max(1, radius - 2), 1)

            if i % marker_stride == 0 or i == n - 1:
                marker_txt = self.font_small.render(
                    f"{traded/1000:.0f}k | {slip:.1f}bp",
                    True,
                    cfg.text_color,
                )
                self.screen.blit(marker_txt, (x_pos + 5, y_pos - 10))

        # Legend
        legend_items = [
            ("Price idx", cfg.price_color),
            ("Exec idx", cfg.exec_price_color),
            ("Inventory", cfg.inventory_color),
            ("Cash idx", cfg.cash_color),
            ("Wealth idx", cfg.optimal_color),
            ("Trade marker: size=volume, color=slippage", (220, 220, 220)),
        ]
        lx = ix + 8
        ly = iy + 6
        for label, color in legend_items:
            if "Trade marker" in label:
                pygame.draw.circle(self.screen, (255, 130, 80), (lx + 6, ly + 7), 4)
                self.screen.blit(self.font_small.render(label, True, cfg.text_color), (lx + 16, ly))
                break
            pygame.draw.line(self.screen, color, (lx, ly + 7), (lx + 14, ly + 7), 2)
            self.screen.blit(self.font_small.render(label, True, cfg.text_color), (lx + 18, ly))
            lx += 106

    def _draw_footer(self, x: int, y: int, w: int, h: int):
        """Draw compact summary stats for the latest step."""
        cfg = self.config
        rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(self.screen, cfg.grid_color, rect, 1)

        if len(self.prices) == 0:
            return

        price0 = max(self.prices[0], 1e-8)
        inv0 = max(self.inventories[0], 1e-8)
        notional0 = max(price0 * inv0, 1e-8)

        cur_price = self.prices[-1]
        cur_exec = self.exec_prices[-1]
        cur_inv = self.inventories[-1]
        cur_cash = self.cash_values[-1]
        cur_wealth = self.wealth_values[-1]
        cur_shortfall = self.shortfalls[-1]
        cur_slip = self.slippages[-1] if len(self.slippages) > 0 else 0.0
        cur_trade = self.shares_traded[-1] if len(self.shares_traded) > 0 else 0.0

        price_idx = cur_price / price0
        inv_ratio = cur_inv / inv0
        cash_idx = cur_cash / notional0
        wealth_idx = cur_wealth / notional0
        shortfall_bps = 10000.0 * cur_shortfall / notional0

        summary = (
            f"Step {max(0, len(self.prices)-1):>2}/{self.max_steps}   "
            f"Price ${cur_price:,.2f} ({price_idx:.3f}x)   "
            f"Exec ${cur_exec:,.2f}   "
            f"Inventory {inv_ratio*100:6.2f}%   "
            f"Cash ${cur_cash:,.0f} ({cash_idx:.3f}x)   "
            f"Wealth {wealth_idx:.3f}x   "
            f"Trade {cur_trade:,.0f}   "
            f"Slippage {cur_slip:+.2f} bp   "
            f"Shortfall {shortfall_bps:+.2f} bp"
        )
        self.screen.blit(self.font_small.render(summary, True, cfg.text_color), (x + 10, y + h // 2 - 8))

    def _slippage_to_color(self, slippage_bps: float) -> Tuple[int, int, int]:
        """Map slippage to marker color."""
        magnitude = min(abs(slippage_bps) / 40.0, 1.0)
        if slippage_bps >= 0:
            # Worse execution -> warmer red/orange.
            r = 255
            g = int(200 - 120 * magnitude)
            b = int(120 - 70 * magnitude)
            return (r, g, b)
        # Better-than-arrival execution -> greener.
        r = int(120 - 60 * magnitude)
        g = int(220 - 20 * magnitude)
        b = int(120 - 40 * magnitude)
        return (r, g, b)

    def _draw_panel_frame(self, x: int, y: int, w: int, h: int, title: str):
        """Draw panel background and title."""
        cfg = self.config

        # Background
        rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(self.screen, cfg.grid_color, rect, 1)

        # Title
        title_surf = self.font.render(title, True, cfg.text_color)
        self.screen.blit(title_surf, (x + 5, y + 2))

        # Return inner area
        return (x + 5, y + 20, w - 10, h - 25)

    def _time_to_x(self, t: float, x_start: int, w: int) -> int:
        """Convert time to x coordinate."""
        # T_max from params would be better, approximate with max_steps
        t_max = len(self.times) / max(1, self.max_steps)
        t_max = max(t_max, 0.1)
        return int(x_start + (t / t_max) * w * 0.9)

    def _step_to_x(self, step: int, x_start: int, w: int) -> int:
        """Convert step index to x coordinate."""
        if len(self.times) <= 1:
            return x_start
        return int(x_start + (step / max(1, len(self.times) - 1)) * w * 0.9)

    def _draw_price_panel(self, x: int, y: int, w: int, h: int):
        """Draw price panel with unaffected and execution prices."""
        cfg = self.config
        ix, iy, iw, ih = self._draw_panel_frame(x, y, w, h, "Price Evolution")

        if len(self.prices) < 2:
            return

        price_min, price_max = self.price_range

        def price_to_y(p: float) -> int:
            return int(iy + ih - (p - price_min) / (price_max - price_min) * ih)

        # Draw grid lines
        for i in range(5):
            gy = iy + int(ih * i / 4)
            pygame.draw.line(self.screen, cfg.grid_color, (ix, gy), (ix + iw, gy), 1)
            price_label = price_max - (price_max - price_min) * i / 4
            label = self.font_small.render(f"${price_label:.2f}", True, cfg.text_color)
            self.screen.blit(label, (ix + iw - 60, gy - 6))

        # Draw price lines
        for i in range(1, len(self.prices)):
            x1 = self._step_to_x(i - 1, ix, iw)
            x2 = self._step_to_x(i, ix, iw)

            # Unaffected price
            y1 = price_to_y(self.prices[i - 1])
            y2 = price_to_y(self.prices[i])
            pygame.draw.line(self.screen, cfg.price_color, (x1, y1), (x2, y2), 2)

            # Execution price (only when trading)
            if self.shares_traded[i] > 0:
                ey = price_to_y(self.exec_prices[i])
                pygame.draw.circle(self.screen, cfg.exec_price_color, (x2, ey), 4)

        # Legend
        pygame.draw.line(self.screen, cfg.price_color, (ix + 10, iy + 5), (ix + 30, iy + 5), 2)
        self.screen.blit(self.font_small.render("Unaffected", True, cfg.text_color), (ix + 35, iy))
        pygame.draw.circle(self.screen, cfg.exec_price_color, (ix + 120, iy + 5), 4)
        self.screen.blit(self.font_small.render("Execution", True, cfg.text_color), (ix + 130, iy))

    def _draw_inventory_panel(self, x: int, y: int, w: int, h: int):
        """Draw inventory panel with actual vs optimal trajectory."""
        cfg = self.config
        ix, iy, iw, ih = self._draw_panel_frame(x, y, w, h, "Inventory Trajectory")

        if len(self.inventories) < 2:
            return

        inv_max = self.inventory_max

        def inv_to_y(inv: float) -> int:
            return int(iy + ih - (inv / inv_max) * ih)

        # Draw grid
        for i in range(5):
            gy = iy + int(ih * i / 4)
            pygame.draw.line(self.screen, cfg.grid_color, (ix, gy), (ix + iw, gy), 1)

        # Draw trajectories
        for i in range(1, len(self.inventories)):
            x1 = self._step_to_x(i - 1, ix, iw)
            x2 = self._step_to_x(i, ix, iw)

            # Optimal
            oy1 = inv_to_y(self.optimal_inventories[i - 1])
            oy2 = inv_to_y(self.optimal_inventories[i])
            pygame.draw.line(self.screen, cfg.optimal_color, (x1, oy1), (x2, oy2), 2)

            # Actual
            ay1 = inv_to_y(self.inventories[i - 1])
            ay2 = inv_to_y(self.inventories[i])
            pygame.draw.line(self.screen, cfg.inventory_color, (x1, ay1), (x2, ay2), 2)

        # Current values
        if len(self.inventories) > 0:
            actual = self.inventories[-1]
            optimal = self.optimal_inventories[-1]
            dev = actual - optimal
            label = f"Actual: {actual:,.0f}  Optimal: {optimal:,.0f}  Dev: {dev:+,.0f}"
            self.screen.blit(self.font_small.render(label, True, cfg.text_color), (ix + iw - 350, iy))

        # Legend
        pygame.draw.line(self.screen, cfg.inventory_color, (ix + 10, iy + 5), (ix + 30, iy + 5), 2)
        self.screen.blit(self.font_small.render("Actual", True, cfg.text_color), (ix + 35, iy))
        pygame.draw.line(self.screen, cfg.optimal_color, (ix + 100, iy + 5), (ix + 120, iy + 5), 2)
        self.screen.blit(self.font_small.render("Optimal", True, cfg.text_color), (ix + 125, iy))

    def _draw_action_panel(self, x: int, y: int, w: int, h: int):
        """Draw trading actions as bar chart with slippage."""
        cfg = self.config
        ix, iy, iw, ih = self._draw_panel_frame(x, y, w, h, "Trading Actions & Slippage")

        if len(self.shares_traded) < 2:
            return

        action_max = self.action_max
        bar_width = max(5, int(iw * 0.8 / max(1, len(self.shares_traded))))

        for i in range(1, len(self.shares_traded)):
            x_pos = self._step_to_x(i, ix, iw)
            bar_h = int((self.shares_traded[i] / action_max) * ih * 0.8)
            bar_y = iy + ih - bar_h

            # Bar for shares traded
            bar_rect = pygame.Rect(x_pos - bar_width//2, bar_y, bar_width, bar_h)
            pygame.draw.rect(self.screen, cfg.action_color, bar_rect)

            # Slippage indicator (red dot size proportional to slippage)
            slippage = abs(self.slippages[i])
            if slippage > 0:
                dot_size = min(10, int(slippage / 5) + 2)
                pygame.draw.circle(self.screen, cfg.slippage_color, (x_pos, bar_y - 5), dot_size)

        # Current action info
        if len(self.shares_traded) > 0:
            last_trade = self.shares_traded[-1]
            last_slip = self.slippages[-1]
            label = f"Last trade: {last_trade:,.0f} shares  Slippage: {last_slip:.1f} bps"
            self.screen.blit(self.font_small.render(label, True, cfg.text_color), (ix + iw - 300, iy))

    def _draw_cash_panel(self, x: int, y: int, w: int, h: int):
        """Draw cash and implementation shortfall."""
        cfg = self.config
        ix, iy, iw, ih = self._draw_panel_frame(x, y, w, h, "Cash & Implementation Shortfall")

        if len(self.cash_values) < 2:
            return

        cash_max = self.cash_max

        def cash_to_y(c: float) -> int:
            return int(iy + ih - (c / cash_max) * ih * 0.9)

        # Draw grid
        for i in range(5):
            gy = iy + int(ih * i / 4)
            pygame.draw.line(self.screen, cfg.grid_color, (ix, gy), (ix + iw, gy), 1)

        # Draw cash line
        for i in range(1, len(self.cash_values)):
            x1 = self._step_to_x(i - 1, ix, iw)
            x2 = self._step_to_x(i, ix, iw)

            y1 = cash_to_y(self.cash_values[i - 1])
            y2 = cash_to_y(self.cash_values[i])
            pygame.draw.line(self.screen, cfg.cash_color, (x1, y1), (x2, y2), 2)

        # Current values
        if len(self.cash_values) > 0:
            cash = self.cash_values[-1]
            shortfall = self.shortfalls[-1]
            # Estimate shortfall in bps
            if len(self.inventories) > 0 and self.inventories[0] > 0:
                initial_notional = self.prices[0] * self.inventories[0]
                shortfall_bps = 10000 * shortfall / initial_notional if initial_notional > 0 else 0
            else:
                shortfall_bps = 0

            label = f"Cash: ${cash:,.0f}  Shortfall: ${shortfall:,.0f} ({shortfall_bps:.1f} bps)"
            self.screen.blit(self.font_small.render(label, True, cfg.text_color), (ix + iw - 400, iy))

    def close(self):
        """Clean up pygame resources."""
        if self.initialized:
            pygame.quit()
            self.screen = None
            self.clock = None
            self.font = None
            self.font_small = None
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
