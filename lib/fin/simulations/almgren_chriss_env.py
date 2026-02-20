"""
Almgren-Chriss Optimal Execution Environment

Single-agent RL environment for optimal trade execution with:
- Linear permanent price impact
- Linear temporary price impact
- Mean-variance or CARA utility objective
- Closed-form optimal strategy for benchmarking

Model:
    Unaffected price: dS̃_t = μ dt + σ dW_t
    Execution price: S_t = S̃_t - g(v_t) - h(x_t)

    where:
        v_t = trading rate (shares/time)
        x_t = cumulative shares traded
        g(v) = η * v (temporary impact)
        h(x) = γ * x (permanent impact)

Objective (mean-variance):
    min E[C] + λ * Var[C]

    where C = implementation shortfall (cost of execution)

References:
    Almgren, R., & Chriss, N. (2001). Optimal execution of portfolio transactions.
    Journal of Risk, 3, 5-40.

Author: Yunian Pan
Email: yp1170@nyu.edu
"""

import numpy as np
from datetime import date
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None
    spaces = None

from ..processes import GBM, SimulationConfig
from ..utils.utility import CARAUtility

# Renderer import (optional)
try:
    from .almgren_chriss_renderer import AlmgrenChrissRenderer, RenderConfig
    RENDERER_AVAILABLE = True
except ImportError:
    RENDERER_AVAILABLE = False
    AlmgrenChrissRenderer = None
    RenderConfig = None


@dataclass
class AlmgrenChrissParams:
    """
    Almgren-Chriss model parameters.

    Price dynamics:
        dS̃_t = μ dt + σ dW_t  (unaffected price)

    Market impact:
        Permanent: γ (price moves by γ per share traded, persists)
        Temporary: η (additional cost per share, doesn't persist)

    Typical calibration (Almgren 2003):
        - γ ≈ 2.5e-7 for large-cap stocks
        - η ≈ 2.5e-6 for large-cap stocks
        - η/γ ≈ 10 (temporary >> permanent for liquid stocks)
    """
    # Initial conditions
    S_0: float = 100.0          # Initial stock price
    X_0: float = 1_000_000      # Initial inventory (shares to liquidate)

    # Price dynamics (unaffected)
    mu: float = 0.0             # Drift (usually 0 for short horizons)
    sigma: float = 0.02         # Daily volatility (2%)

    # Market impact parameters
    gamma: float = 2.5e-7       # Permanent impact ($/share)
    eta: float = 2.5e-6         # Temporary impact ($/share per share/day)

    # Execution horizon
    T: float = 1.0              # Trading horizon (in trading days)

    # Risk aversion
    lambda_var: float = 1e-6    # Mean-variance risk aversion

    def __post_init__(self):
        """Validate parameters."""
        assert self.S_0 > 0, "Initial price must be positive"
        assert self.X_0 > 0, "Initial inventory must be positive"
        assert self.sigma >= 0, "Volatility must be non-negative"
        assert self.gamma >= 0, "Permanent impact must be non-negative"
        assert self.eta >= 0, "Temporary impact must be non-negative"
        assert self.T > 0, "Time horizon must be positive"


class AlmgrenChrissEnv(gym.Env if gym else object):
    """
    Almgren-Chriss Optimal Execution Environment.

    The agent must liquidate X_0 shares over time horizon T while minimizing
    execution cost (implementation shortfall).

    State:
        - S_t: Current (unaffected) stock price
        - q_t: Remaining inventory (shares)
        - t: Current time (or time remaining T - t)

    Action:
        - n_t: Number of shares to trade this period
          (continuous: any value in [0, q_t])
          (discrete: fraction of remaining inventory)

    Reward:
        - Cash received minus implementation shortfall
        - Or CARA utility of execution proceeds

    Terminal:
        - Episode ends when t >= T or q_t <= 0
        - Forced liquidation of remaining shares at T
    """

    metadata = {'render_modes': ['human', 'ansi', 'pygame', 'rgb_array'], 'render_fps': 30}

    def __init__(
        self,
        params: Optional[AlmgrenChrissParams] = None,
        n_steps: int = 20,                      # Number of trading periods
        action_type: str = 'continuous',        # 'continuous' or 'discrete'
        n_actions: int = 11,                    # For discrete: 0%, 10%, ..., 100%
        reward_type: str = 'pnl',               # 'pnl', 'shortfall', 'cara'
        cara_gamma: float = 1e-6,               # CARA risk aversion (if reward_type='cara')
        normalize_obs: bool = True,             # Normalize observations
        normalize_reward: bool = True,          # Scale rewards
        penalize_early_liquidation: bool = False,  # Penalty for finishing early
        render_mode: Optional[str] = None,
    ):
        """
        Initialize Almgren-Chriss environment.

        Args:
            params: Model parameters (uses defaults if None)
            n_steps: Number of trading periods in horizon T
            action_type: 'continuous' (trade any amount) or 'discrete' (fixed fractions)
            n_actions: Number of discrete actions (if action_type='discrete')
            reward_type:
                'pnl': Raw P&L from trades
                'shortfall': Negative implementation shortfall
                'cara': CARA utility of proceeds
            cara_gamma: Risk aversion for CARA utility
            normalize_obs: Normalize observations to [0, 1] range
            normalize_reward: Scale rewards by initial notional
            penalize_early_liquidation: Add penalty for finishing before T
            render_mode: 'human' for console output
        """
        super().__init__()

        self.params = params if params is not None else AlmgrenChrissParams()
        self.n_steps = n_steps
        self.dt = self.params.T / n_steps
        self.action_type = action_type
        self.n_actions = n_actions
        self.reward_type = reward_type
        self.normalize_obs = normalize_obs
        self.normalize_reward = normalize_reward
        self.penalize_early_liquidation = penalize_early_liquidation
        self.render_mode = render_mode

        # CARA utility for risk-sensitive reward
        if reward_type == 'cara':
            self.utility = CARAUtility(gamma=cara_gamma)
        else:
            self.utility = None

        # Create GBM process for unaffected price
        self.price_process = GBM(mu=self.params.mu, sigma=self.params.sigma)

        # Action space
        if action_type == 'continuous':
            # Action: fraction of remaining inventory to trade [0, 1]
            self.action_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(1,),
                dtype=np.float32
            )
        else:
            # Discrete actions: 0%, 10%, 20%, ..., 100% of remaining
            self.action_space = spaces.Discrete(n_actions)

        # Observation space: [price, inventory, time]
        # Normalized to approximately [0, 1] range
        if normalize_obs:
            self.observation_space = spaces.Box(
                low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([2.0, 1.0, 1.0], dtype=np.float32),  # Price can exceed 2x
                dtype=np.float32
            )
        else:
            self.observation_space = spaces.Box(
                low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([np.inf, self.params.X_0, self.params.T], dtype=np.float32),
                dtype=np.float32
            )

        # State variables
        self.S = None           # Current unaffected price
        self.q = None           # Remaining inventory
        self.t = None           # Current time
        self.step_count = None  # Step counter
        self.cumulative_traded = None  # Total shares traded (for permanent impact)
        self.cash = None        # Cash from sales
        self.arrival_price = None  # Price at start (for shortfall calc)

        # Pre-simulated price path (optional, for consistency)
        self.price_path = None

        # History for analysis
        self.history = {
            'S': [],
            'q': [],
            't': [],
            'action': [],
            'shares_traded': [],
            'execution_price': [],
            'cash': [],
            'reward': [],
        }

        # Pygame renderer (for 'pygame' or 'rgb_array' mode)
        self.renderer = None
        if render_mode in ['pygame', 'rgb_array'] and RENDERER_AVAILABLE:
            self.renderer = AlmgrenChrissRenderer(max_steps=n_steps)

        # Compute optimal strategy for benchmarking
        self._compute_optimal_strategy()

    def _compute_optimal_strategy(self):
        """
        Compute closed-form optimal Almgren-Chriss strategy.

        For mean-variance objective with linear impact:
            x*(t) = X_0 * sinh(κ(T-t)) / sinh(κT)

        where:
            κ = sqrt(λ * σ² / η)

        Trading rate:
            v*(t) = -dx*/dt = X_0 * κ * cosh(κ(T-t)) / sinh(κT)

        Special cases:
            - λ → 0 (risk-neutral): TWAP (uniform liquidation)
            - λ → ∞ (infinitely risk-averse): immediate liquidation
        """
        p = self.params

        # Risk-adjusted parameter
        if p.lambda_var > 0 and p.eta > 0:
            self.kappa = np.sqrt(p.lambda_var * p.sigma**2 / p.eta)
        else:
            self.kappa = 0.0

        # Store for trajectory computation
        self.optimal_trajectory_computed = True

    def optimal_inventory(self, t: float) -> float:
        """
        Optimal remaining inventory at time t.

        x*(t) = X_0 * sinh(κ(T-t)) / sinh(κT)

        Args:
            t: Current time

        Returns:
            Optimal inventory at time t
        """
        p = self.params

        if self.kappa < 1e-10:
            # Risk-neutral: TWAP
            return p.X_0 * (1 - t / p.T)
        else:
            # Risk-averse: front-loaded
            return p.X_0 * np.sinh(self.kappa * (p.T - t)) / np.sinh(self.kappa * p.T)

    def optimal_trade_rate(self, t: float) -> float:
        """
        Optimal trading rate at time t.

        v*(t) = X_0 * κ * cosh(κ(T-t)) / sinh(κT)

        Args:
            t: Current time

        Returns:
            Optimal shares per unit time at time t
        """
        p = self.params

        if self.kappa < 1e-10:
            # Risk-neutral: TWAP
            return p.X_0 / p.T
        else:
            # Risk-averse
            return p.X_0 * self.kappa * np.cosh(self.kappa * (p.T - t)) / np.sinh(self.kappa * p.T)

    def optimal_shares_this_period(self, t: float, dt: float) -> float:
        """
        Optimal shares to trade in period [t, t+dt].

        Approximation: v*(t) * dt

        Or exact: x*(t) - x*(t+dt)
        """
        return self.optimal_inventory(t) - self.optimal_inventory(t + dt)

    def expected_cost(self) -> Tuple[float, float]:
        """
        Expected cost and variance of optimal strategy.

        E[C] = γ * X_0² / 2 + η * X_0² * κ / (2 * tanh(κT))

        Var[C] = σ² * X_0² / (2κ) * tanh(κT/2)  [approximately]

        Returns:
            Tuple of (expected_cost, cost_variance)
        """
        p = self.params
        X = p.X_0

        # Permanent impact cost (unavoidable)
        perm_cost = 0.5 * p.gamma * X**2

        if self.kappa < 1e-10:
            # TWAP
            temp_cost = p.eta * X**2 / (2 * p.T)
            var_cost = (p.sigma**2 * X**2 * p.T) / 3
        else:
            # Risk-averse strategy
            kT = self.kappa * p.T
            temp_cost = p.eta * X**2 * self.kappa / (2 * np.tanh(kT))
            var_cost = (p.sigma**2 * X**2 / (2 * self.kappa)) * np.tanh(kT / 2)

        return perm_cost + temp_cost, var_cost

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        p = self.params

        # Initialize state
        self.S = p.S_0
        self.q = p.X_0
        self.t = 0.0
        self.step_count = 0
        self.cumulative_traded = 0.0
        self.cash = 0.0
        self.arrival_price = p.S_0

        # Pre-simulate price path for this episode
        config = SimulationConfig(n_paths=1, n_steps=self.n_steps, random_seed=seed)
        X0 = np.array([[p.S_0]])
        t_grid, paths = self.price_process.simulate(X0, p.T, config, scheme='exact')
        self.price_path = paths[:, 0, 0]  # Shape: (n_steps+1,)
        self.t_grid = t_grid

        # Clear history
        self.history = {k: [] for k in self.history.keys()}
        self._record_state(action=0.0, shares=0.0, exec_price=0.0, reward=0.0)

        # Reset renderer
        if self.renderer is not None:
            self.renderer.reset()
            self.renderer.initialize(interactive=(self.render_mode == 'pygame'))
            # Initial state
            self.renderer.update(
                t=self.t,
                price=self.S,
                exec_price=self.S,
                inventory=self.q,
                optimal_inventory=self.optimal_inventory(self.t),
                shares_traded=0,
                cash=self.cash,
                shortfall=0,
                arrival_price=self.arrival_price,
            )

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one trading period.

        Args:
            action: Trading decision
                - continuous: fraction of remaining inventory [0, 1]
                - discrete: action index (maps to fraction)

        Returns:
            observation, reward, terminated, truncated, info
        """
        p = self.params

        # Parse action to shares traded
        if self.action_type == 'continuous':
            fraction = np.clip(action, 0.0, 1.0).item()
        else:
            fraction = action / (self.n_actions - 1)

        shares_to_trade = fraction * self.q
        shares_to_trade = min(shares_to_trade, self.q)  # Can't trade more than we have

        # Current unaffected price (from pre-simulated path)
        S_unaffected = self.price_path[self.step_count]

        # Compute execution price with market impact
        # Permanent impact: price moves down by γ per share already traded
        permanent_impact = p.gamma * self.cumulative_traded

        # Temporary impact: additional cost for trading rate
        trading_rate = shares_to_trade / self.dt if self.dt > 0 else 0
        temporary_impact = p.eta * trading_rate

        # Execution price
        execution_price = S_unaffected - permanent_impact - temporary_impact
        execution_price = max(execution_price, 0.01)  # Floor at small positive

        # Execute trade
        trade_proceeds = shares_to_trade * execution_price
        self.cash += trade_proceeds
        self.q -= shares_to_trade
        self.cumulative_traded += shares_to_trade

        # Advance time
        self.step_count += 1
        self.t += self.dt

        # Update price from path
        if self.step_count < len(self.price_path):
            self.S = self.price_path[self.step_count]

        # Compute reward for this step's trade
        reward = self._compute_reward(shares_to_trade, execution_price)

        # Check termination
        terminated = False
        truncated = False

        # Episode ends when time horizon reached or fully liquidated
        if self.step_count >= self.n_steps:
            truncated = True
            # Force liquidate any remaining inventory at terminal price
            if self.q > 0:
                terminal_price = self.S - p.gamma * self.cumulative_traded - p.eta * (self.q / self.dt)
                terminal_price = max(terminal_price, 0.01)
                # Add terminal liquidation reward/cost to total reward
                terminal_reward = self._compute_reward(self.q, terminal_price)
                reward += terminal_reward
                self.cash += self.q * terminal_price
                self.q = 0

        if self.q <= 0:
            terminated = True

        # Record history
        self._record_state(
            action=fraction,
            shares=shares_to_trade,
            exec_price=execution_price,
            reward=reward
        )

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _compute_reward(self, shares_traded: float, execution_price: float) -> float:
        """
        Compute step reward based on reward_type.

        Args:
            shares_traded: Shares traded this step
            execution_price: Price received per share

        Returns:
            Reward value
        """
        p = self.params

        # Normalize factor
        norm = (p.S_0 * p.X_0) if self.normalize_reward else 1.0

        if self.reward_type == 'pnl':
            # Simple P&L: cash received
            reward = shares_traded * execution_price / norm

        elif self.reward_type == 'shortfall':
            # Implementation shortfall: compare to arrival price
            # Shortfall = (arrival_price - execution_price) * shares
            shortfall = (self.arrival_price - execution_price) * shares_traded
            reward = -shortfall / norm  # Negative shortfall = good

        elif self.reward_type == 'cara':
            # CARA utility (computed at episode end)
            # For step reward, use incremental cash
            reward = shares_traded * execution_price / norm

        else:
            reward = shares_traded * execution_price / norm

        # Optional penalty for early liquidation
        if self.penalize_early_liquidation and self.q <= 0 and self.t < p.T:
            time_remaining = p.T - self.t
            reward -= 0.1 * time_remaining / p.T  # Small penalty

        return float(reward)

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        p = self.params

        if self.normalize_obs:
            obs = np.array([
                self.S / p.S_0,           # Normalized price
                self.q / p.X_0,           # Fraction of inventory remaining
                self.t / p.T,             # Fraction of time elapsed
            ], dtype=np.float32)
        else:
            obs = np.array([
                self.S,
                self.q,
                self.t,
            ], dtype=np.float32)

        return obs

    def _get_info(self) -> Dict[str, Any]:
        """Get info dict with additional metrics."""
        p = self.params

        # Implementation shortfall so far
        ideal_value = self.arrival_price * (p.X_0 - self.q)
        shortfall = ideal_value - self.cash

        # Optimal trajectory comparison
        optimal_q = self.optimal_inventory(self.t)
        inventory_deviation = self.q - optimal_q

        info = {
            'price': self.S,
            'inventory': self.q,
            'initial_inventory': p.X_0,
            'time': self.t,
            'cash': self.cash,
            'cumulative_traded': self.cumulative_traded,
            'shortfall': shortfall,
            'shortfall_bps': 10000 * shortfall / (p.S_0 * p.X_0) if p.X_0 > 0 else 0,
            'optimal_inventory': optimal_q,
            'inventory_deviation': inventory_deviation,
            'step': self.step_count,
            'remaining_steps': self.n_steps - self.step_count,
            'n_steps': self.n_steps,
        }

        return info

    def _record_state(self, action: float, shares: float, exec_price: float, reward: float):
        """Record state to history."""
        self.history['S'].append(self.S)
        self.history['q'].append(self.q)
        self.history['t'].append(self.t)
        self.history['action'].append(action)
        self.history['shares_traded'].append(shares)
        self.history['execution_price'].append(exec_price)
        self.history['cash'].append(self.cash)
        self.history['reward'].append(reward)

    def render(self):
        """Render current state."""
        if self.render_mode == 'human':
            p = self.params
            print(f"\nStep {self.step_count}/{self.n_steps} (t={self.t:.3f}/{p.T:.3f})")
            print(f"  Price: ${self.S:.2f} (arrival: ${self.arrival_price:.2f})")
            print(f"  Inventory: {self.q:,.0f} / {p.X_0:,.0f} shares ({100*self.q/p.X_0:.1f}%)")
            print(f"  Cash: ${self.cash:,.2f}")

            # Compare to optimal
            optimal_q = self.optimal_inventory(self.t)
            print(f"  Optimal inventory: {optimal_q:,.0f} (dev: {self.q - optimal_q:+,.0f})")

            # Implementation shortfall
            info = self._get_info()
            print(f"  Shortfall: ${info['shortfall']:,.2f} ({info['shortfall_bps']:.1f} bps)")
            print("-" * 50)

        elif self.render_mode == 'ansi':
            return f"t={self.t:.2f} S=${self.S:.2f} q={self.q:.0f} cash=${self.cash:.2f}"

        elif self.render_mode in ['pygame', 'rgb_array']:
            if self.renderer is not None:
                # Update renderer with latest state
                info = self._get_info()
                last_shares = self.history['shares_traded'][-1] if self.history['shares_traded'] else 0
                if last_shares > 0 and self.history['execution_price']:
                    last_exec_price = self.history['execution_price'][-1]
                else:
                    # Avoid fake spike from reset placeholder exec_price=0.
                    last_exec_price = self.S

                self.renderer.update(
                    t=self.t,
                    price=self.S,
                    exec_price=last_exec_price,
                    inventory=self.q,
                    optimal_inventory=info['optimal_inventory'],
                    shares_traded=last_shares,
                    cash=self.cash,
                    shortfall=info['shortfall'],
                    arrival_price=self.arrival_price,
                )

                # Render and return RGB array if requested
                return self.renderer.render()

    def get_optimal_trajectory(self) -> Dict[str, np.ndarray]:
        """
        Get optimal trajectory for comparison.

        Returns:
            Dict with 't', 'inventory', 'trade_rate' arrays
        """
        t_vals = np.linspace(0, self.params.T, self.n_steps + 1)
        inventory = np.array([self.optimal_inventory(t) for t in t_vals])
        trade_rate = np.array([self.optimal_trade_rate(t) for t in t_vals])

        return {
            't': t_vals,
            'inventory': inventory,
            'trade_rate': trade_rate,
        }

    def get_history(self) -> Dict[str, list]:
        """Get episode history."""
        return self.history

    def close(self):
        """Clean up."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


def make_almgren_chriss_env(
    # Initial conditions
    S_0: float = 100.0,
    X_0: float = 1_000_000,
    # Dynamics
    sigma: float = 0.02,
    # Impact
    gamma: float = 2.5e-7,
    eta: float = 2.5e-6,
    # Horizon
    T: float = 1.0,
    n_steps: int = 20,
    # Risk aversion
    lambda_var: float = 1e-6,
    # Env settings
    action_type: str = 'continuous',
    reward_type: str = 'shortfall',
    mu: float = 0.0,
    **kwargs
) -> AlmgrenChrissEnv:
    """
    Create Almgren-Chriss environment with custom parameters.

    Args:
        S_0: Initial stock price
        X_0: Shares to liquidate
        sigma: Daily volatility
        gamma: Permanent impact parameter
        eta: Temporary impact parameter
        T: Trading horizon (days)
        n_steps: Number of trading periods
        lambda_var: Mean-variance risk aversion
        action_type: 'continuous' or 'discrete'
        reward_type: 'pnl', 'shortfall', or 'cara'
        mu: Drift per env time unit
        **kwargs: Additional AlmgrenChrissEnv arguments

    Returns:
        Configured environment

    Examples:
        >>> # Default liquid stock
        >>> env = make_almgren_chriss_env()

        >>> # Illiquid stock with higher impact
        >>> env = make_almgren_chriss_env(
        ...     gamma=1e-6,
        ...     eta=1e-5,
        ...     lambda_var=1e-5,  # More risk averse
        ... )

        >>> # Quick execution
        >>> env = make_almgren_chriss_env(T=0.5, n_steps=10)
    """
    params = AlmgrenChrissParams(
        S_0=S_0,
        X_0=X_0,
        mu=mu,
        sigma=sigma,
        gamma=gamma,
        eta=eta,
        T=T,
        lambda_var=lambda_var,
    )

    return AlmgrenChrissEnv(
        params=params,
        n_steps=n_steps,
        action_type=action_type,
        reward_type=reward_type,
        **kwargs
    )


def make_almgren_chriss_env_with_ticker(
    ticker: str,
    model: str = "gbm",
    history_days: int = 252,
    periods_per_year: int = 252,
    env_time_scale: str = "daily",
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    risk_free_rate: float = 0.05,
    dividend_yield: float = 0.0,
    history_interval: str = "1d",
    sabr_beta: Optional[float] = 0.5,
    heston_method: str = "L-BFGS-B",
    heston_maxiter: int = 250,
    heston_tol: float = 1e-6,
    fetcher: Optional[Any] = None,
    # AC env params
    S_0: Optional[float] = None,
    X_0: float = 1_000_000,
    gamma: float = 2.5e-7,
    eta: float = 2.5e-6,
    T: float = 1.0,
    n_steps: int = 20,
    lambda_var: float = 1e-6,
    action_type: str = "continuous",
    reward_type: str = "shortfall",
    return_calibration: bool = False,
    **kwargs,
):
    """
    Build single-agent AC env by calibrating dynamics from a market ticker.

    Notes:
        - `gbm` only uses historical closes.
        - `sabr`/`heston` use option chain for sigma and historical drift for mu.
    """
    from .ac_calibration import calibrate_ac_dynamics_from_ticker

    calibration = calibrate_ac_dynamics_from_ticker(
        ticker=ticker,
        model=model,
        history_days=history_days,
        periods_per_year=periods_per_year,
        env_time_scale=env_time_scale,
        start_date=start_date,
        end_date=end_date,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        history_interval=history_interval,
        sabr_beta=sabr_beta,
        heston_method=heston_method,
        heston_maxiter=heston_maxiter,
        heston_tol=heston_tol,
        fetcher=fetcher,
    )

    env = make_almgren_chriss_env(
        S_0=float(calibration.spot_price if S_0 is None else S_0),
        X_0=X_0,
        mu=calibration.mu_env,
        sigma=calibration.sigma_env,
        gamma=gamma,
        eta=eta,
        T=T,
        n_steps=n_steps,
        lambda_var=lambda_var,
        action_type=action_type,
        reward_type=reward_type,
        **kwargs,
    )
    env.calibration_metadata = calibration

    if return_calibration:
        return env, calibration
    return env


# Local alias for convenience inside this module namespace.
make_env_with_ticker = make_almgren_chriss_env_with_ticker
