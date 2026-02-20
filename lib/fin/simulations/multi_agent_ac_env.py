"""
Multi-Agent Almgren-Chriss Optimal Execution Environment (POSG)

A partially observable stochastic game where multiple agents simultaneously
liquidate their inventories, competing through shared price impact.

Key Features:
- N agents each liquidating their own inventory
- Shared permanent price impact (all trades move the price)
- Private temporary impact (each agent pays their own)
- Partial observability: agents don't see others' inventories
- PettingZoo ParallelEnv API for simultaneous actions

Model:
    Unaffected price: dS̃_t = μ dt + σ dW_t

    Permanent impact (shared):
        S_t = S̃_t - γ * Σᵢ xᵢ(t)  where xᵢ(t) = cumulative trades by agent i

    Execution price for agent i:
        Sᵢ_exec = S_t - ηᵢ * vᵢ(t)  where vᵢ(t) = trading rate of agent i

Partial Observability:
    Each agent observes:
        - Current price S_t (affected by all agents' past trades)
        - Own inventory qᵢ
        - Own cumulative trades xᵢ
        - Time remaining T - t
        - (Optional) Aggregate market volume

    Hidden from each agent:
        - Other agents' inventories qⱼ (j ≠ i)
        - Other agents' trading rates vⱼ
        - Individual contributions to price impact

References:
    Carlin, B. I., Lobo, M. S., & Viswanathan, S. (2007).
    Episodic liquidity crises: Cooperative and predatory trading.
    The Journal of Finance, 62(5), 2235-2274.

Author: Yunian Pan
Email: yp1170@nyu.edu
"""

from __future__ import annotations

import functools
import numpy as np
from datetime import date
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from gymnasium import spaces

try:
    from pettingzoo import ParallelEnv
    from pettingzoo.utils import wrappers, parallel_to_aec
    PETTINGZOO_AVAILABLE = True
except ImportError:
    ParallelEnv = object
    PETTINGZOO_AVAILABLE = False
    wrappers = None
    parallel_to_aec = None

from ..processes import GBM, SimulationConfig

try:
    from .multi_agent_ac_renderer import MultiAgentACRenderer, PYGAME_AVAILABLE
except ImportError:
    PYGAME_AVAILABLE = False
    MultiAgentACRenderer = None


@dataclass
class MultiAgentACParams:
    """
    Parameters for Multi-Agent Almgren-Chriss environment.

    Agents can have heterogeneous:
    - Initial inventories
    - Risk aversion (for computing their individual optimal strategies)
    - Temporary impact parameters (private trading costs)

    Partial Observability:
        The environment supports configurable partial observability to test
        approximate information state algorithms. Each observation component
        can be:
        - Fully visible (default)
        - Hidden (not in observation)
        - Noisy (Gaussian noise added)

        This creates a POSG where agents must maintain beliefs about hidden
        state components based on their action history.
    """
    # Price dynamics
    S_0: float = 100.0              # Initial stock price
    mu: float = 0.0                 # Price drift
    sigma: float = 0.02             # Daily volatility

    # Number of agents
    n_agents: int = 2

    # Initial inventories (per agent, or single value for all)
    X_0: float = 500_000            # Default inventory per agent
    X_0_list: Optional[List[float]] = None  # Heterogeneous inventories

    # Market impact parameters
    gamma: float = 2.5e-7           # Permanent impact (shared)
    eta: float = 2.5e-6             # Temporary impact (default, per agent)
    eta_list: Optional[List[float]] = None  # Heterogeneous temporary impact

    # Execution horizon
    T: float = 1.0                  # Trading horizon (days)

    # Risk aversion (for computing individual optimal strategies)
    lambda_var: float = 1e-6        # Default risk aversion
    lambda_var_list: Optional[List[float]] = None  # Heterogeneous risk aversion

    # =========================================================================
    # PARTIAL OBSERVABILITY SETTINGS
    # =========================================================================
    # Each component can be: visible (True), hidden (False), or noisy (noise_std > 0)

    # Price observation
    observe_price: bool = True              # Can agents see the price?
    price_noise_std: float = 0.0            # Gaussian noise on price (as fraction of S_0)

    # Own inventory observation
    observe_own_inventory: bool = True      # Can agents see their own inventory?
    inventory_noise_std: float = 0.0        # Gaussian noise on inventory (as fraction)

    # Own cumulative trades
    observe_own_cumulative: bool = True     # See own cumulative trades?

    # Time observation
    observe_time: bool = True               # Can agents see elapsed time?

    # Aggregate market info (reveals info about other agents)
    observe_aggregate_volume: bool = False  # See total market volume?
    observe_num_agents: bool = False        # Know how many agents exist?

    # Last action feedback
    observe_last_execution_price: bool = False  # See price received last trade?
    observe_last_trade_size: bool = False       # See own last trade size?

    # Legacy compatibility
    observe_price_impact: bool = True       # Deprecated, use observe_price

    def __post_init__(self):
        """Validate and expand parameters."""
        assert self.n_agents >= 1, "Need at least 1 agent"
        assert self.S_0 > 0, "Initial price must be positive"
        assert self.T > 0, "Time horizon must be positive"
        assert self.price_noise_std >= 0, "Noise std must be non-negative"
        assert self.inventory_noise_std >= 0, "Noise std must be non-negative"

        # Expand heterogeneous parameters
        if self.X_0_list is None:
            self.X_0_list = [self.X_0] * self.n_agents
        if self.eta_list is None:
            self.eta_list = [self.eta] * self.n_agents
        if self.lambda_var_list is None:
            self.lambda_var_list = [self.lambda_var] * self.n_agents

        assert len(self.X_0_list) == self.n_agents
        assert len(self.eta_list) == self.n_agents
        assert len(self.lambda_var_list) == self.n_agents

    def get_obs_dim(self) -> int:
        """Calculate observation dimension based on observability settings."""
        dim = 0
        if self.observe_price:
            dim += 1
        if self.observe_own_inventory:
            dim += 1
        if self.observe_own_cumulative:
            dim += 1
        if self.observe_time:
            dim += 1
        if self.observe_aggregate_volume:
            dim += 1
        if self.observe_num_agents:
            dim += 1
        if self.observe_last_execution_price:
            dim += 1
        if self.observe_last_trade_size:
            dim += 1
        return max(dim, 1)  # At least 1D observation


class MultiAgentAlmgrenChrissEnv(ParallelEnv):
    """
    Multi-Agent Almgren-Chriss Optimal Execution Environment.

    PettingZoo ParallelEnv: all agents act simultaneously each step.

    This is a POSG (Partially Observable Stochastic Game) where:
    - Agents compete through shared price impact
    - Each agent only observes their own state
    - Strategic interaction arises from common price dynamics
    """

    metadata = {
        "name": "multi_agent_almgren_chriss_v0",
        "render_modes": ["human", "ansi", "pygame"],
        "is_parallelizable": True,
    }

    def __init__(
        self,
        params: Optional[MultiAgentACParams] = None,
        n_steps: int = 20,
        reward_type: str = 'shortfall',     # 'pnl', 'shortfall'
        normalize_obs: bool = True,
        normalize_reward: bool = True,
        render_mode: Optional[str] = None,
        info_mode: str = 'full',            # 'full' or 'partial'
    ):
        """
        Initialize multi-agent environment.

        Args:
            params: Environment parameters
            n_steps: Number of trading periods
            reward_type: 'pnl' or 'shortfall'
            normalize_obs: Normalize observations
            normalize_reward: Scale rewards
            render_mode: 'human' for console output
            info_mode: 'full' returns all state in info (for debugging),
                      'partial' respects observability settings (prevents leakage)
        """
        super().__init__()

        if not PETTINGZOO_AVAILABLE:
            raise ImportError("pettingzoo is required. Install with: pip install pettingzoo")

        self.params = params if params is not None else MultiAgentACParams()
        self.n_steps = n_steps
        self.dt = self.params.T / n_steps
        self.reward_type = reward_type
        self.normalize_obs = normalize_obs
        self.normalize_reward = normalize_reward
        self.render_mode = render_mode
        self.info_mode = info_mode

        # Agent setup
        self.possible_agents = [f"trader_{i}" for i in range(self.params.n_agents)]
        self.agents = list(self.possible_agents)

        # Map agent names to indices
        self.agent_name_to_idx = {name: i for i, name in enumerate(self.possible_agents)}

        # Create GBM for unaffected price
        self.price_process = GBM(mu=self.params.mu, sigma=self.params.sigma)

        # Observation space per agent (configurable partial observability)
        obs_dim = self.params.get_obs_dim()

        self._obs_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Action space: fraction of remaining inventory to trade [0, 1]
        self._act_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        # State variables (initialized in reset)
        self.S = None                   # Current unaffected price
        self.S_impacted = None          # Price with permanent impact
        self.inventories = None         # Per-agent inventories
        self.cumulative_traded = None   # Per-agent cumulative trades
        self.cash = None                # Per-agent cash
        self.arrival_price = None       # Price at start
        self.t = None                   # Current time
        self.step_count = None
        self.price_path = None          # Pre-simulated price path

        # Per-agent tracking for partial observability
        self.last_execution_prices = None   # Last execution price per agent
        self.last_trade_sizes = None        # Last trade size per agent

        # Episode tracking
        self.total_market_volume = 0.0  # Aggregate volume this episode

        # History
        self.history = {
            'S': [],
            'S_impacted': [],
            'inventories': [],
            'actions': [],
            'rewards': [],
            't': [],
        }

        # Pygame renderer
        self.renderer = None
        if render_mode == 'pygame':
            if not PYGAME_AVAILABLE or MultiAgentACRenderer is None:
                raise ImportError("pygame is required for pygame rendering. Install with: pip install pygame")
            self.renderer = MultiAgentACRenderer(
                n_agents=self.params.n_agents,
                max_steps=n_steps,
                agent_names=self.possible_agents,
            )

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Space:
        """Get observation space for an agent."""
        return self._obs_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Space:
        """Get action space for an agent."""
        return self._act_space

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict]]:
        """
        Reset environment.

        Returns:
            observations: Dict mapping agent -> observation
            infos: Dict mapping agent -> info dict
        """
        if seed is not None:
            np.random.seed(seed)

        p = self.params

        # Reset agents list
        self.agents = list(self.possible_agents)

        # Initialize state
        self.S = p.S_0
        self.S_impacted = p.S_0
        self.arrival_price = p.S_0
        self.t = 0.0
        self.step_count = 0
        self.total_market_volume = 0.0

        # Per-agent state
        self.inventories = np.array(p.X_0_list, dtype=np.float64)
        self.cumulative_traded = np.zeros(p.n_agents, dtype=np.float64)
        self.cash = np.zeros(p.n_agents, dtype=np.float64)

        # Per-agent tracking for partial observability
        self.last_execution_prices = np.full(p.n_agents, p.S_0, dtype=np.float64)
        self.last_trade_sizes = np.zeros(p.n_agents, dtype=np.float64)

        # Pre-simulate price path
        config = SimulationConfig(n_paths=1, n_steps=self.n_steps, random_seed=seed)
        X0 = np.array([[p.S_0]])
        t_grid, paths = self.price_process.simulate(X0, p.T, config, scheme='exact')
        self.price_path = paths[:, 0, 0]
        self.t_grid = t_grid

        # Clear history
        self.history = {k: [] for k in self.history.keys()}

        # Reset renderer
        if self.renderer is not None:
            self.renderer.reset()

        # Build observations
        observations = {agent: self._get_observation(agent) for agent in self.agents}
        infos = {agent: self._get_info(agent, respect_partial_obs=(self.info_mode == 'partial')) for agent in self.agents}

        return observations, infos

    def step(
        self,
        actions: Dict[str, np.ndarray],
    ) -> Tuple[
        Dict[str, np.ndarray],  # observations
        Dict[str, float],       # rewards
        Dict[str, bool],        # terminations
        Dict[str, bool],        # truncations
        Dict[str, Dict],        # infos
    ]:
        """
        Execute one step with all agents acting simultaneously.

        Args:
            actions: Dict mapping agent -> action array

        Returns:
            observations, rewards, terminations, truncations, infos
        """
        p = self.params

        # Parse actions and compute trades
        shares_traded = np.zeros(p.n_agents, dtype=np.float64)

        for agent, action in actions.items():
            idx = self.agent_name_to_idx[agent]

            # Action is fraction of remaining inventory
            fraction = np.clip(action, 0.0, 1.0).item()
            shares = fraction * self.inventories[idx]
            shares_traded[idx] = shares

        # Update cumulative trades (for permanent impact)
        self.cumulative_traded += shares_traded
        total_shares_this_step = np.sum(shares_traded)
        self.total_market_volume += total_shares_this_step

        # Compute execution prices for each agent
        # Permanent impact: affects the base price for everyone
        total_cumulative = np.sum(self.cumulative_traded)
        permanent_impact = p.gamma * total_cumulative

        # Get unaffected price at this step
        S_unaffected = self.price_path[self.step_count]
        self.S = S_unaffected

        # Impacted price (common to all)
        self.S_impacted = S_unaffected - permanent_impact

        # Compute rewards
        rewards = {}
        for agent in self.agents:
            idx = self.agent_name_to_idx[agent]

            # Temporary impact: private to each agent
            trading_rate = shares_traded[idx] / self.dt if self.dt > 0 else 0
            temporary_impact = p.eta_list[idx] * trading_rate

            # Execution price for this agent
            exec_price = self.S_impacted - temporary_impact
            exec_price = max(exec_price, 0.01)

            # Cash from trade
            trade_proceeds = shares_traded[idx] * exec_price
            self.cash[idx] += trade_proceeds

            # Update inventory
            self.inventories[idx] -= shares_traded[idx]

            # Track for partial observability
            self.last_execution_prices[idx] = exec_price
            self.last_trade_sizes[idx] = shares_traded[idx]

            # Compute reward
            rewards[agent] = self._compute_reward(idx, shares_traded[idx], exec_price)

        # Advance time
        self.step_count += 1
        self.t += self.dt

        # Check termination
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}

        if self.step_count >= self.n_steps:
            # Force liquidate remaining inventory
            for agent in self.agents:
                idx = self.agent_name_to_idx[agent]
                if self.inventories[idx] > 0:
                    # Liquidate at current impacted price minus temporary impact
                    remaining = self.inventories[idx]
                    trading_rate = remaining / self.dt if self.dt > 0 else remaining
                    temp_impact = p.eta_list[idx] * trading_rate
                    final_price = max(self.S_impacted - temp_impact, 0.01)
                    # Add terminal liquidation reward/cost
                    terminal_reward = self._compute_reward(idx, remaining, final_price)
                    rewards[agent] += terminal_reward
                    self.cash[idx] += remaining * final_price
                    self.inventories[idx] = 0

                truncations[agent] = True

        # Check if any agent finished
        for agent in self.agents:
            idx = self.agent_name_to_idx[agent]
            if self.inventories[idx] <= 0:
                terminations[agent] = True

        # Record history
        self.history['S'].append(self.S)
        self.history['S_impacted'].append(self.S_impacted)
        self.history['inventories'].append(self.inventories.copy())
        self.history['actions'].append(shares_traded.copy())
        self.history['rewards'].append(rewards.copy())
        self.history['t'].append(self.t)

        # Build outputs
        observations = {agent: self._get_observation(agent) for agent in self.agents}
        infos = {agent: self._get_info(agent, respect_partial_obs=(self.info_mode == 'partial')) for agent in self.agents}

        # Remove terminated agents (PettingZoo convention)
        # Actually, for ParallelEnv we keep all agents until episode ends

        return observations, rewards, terminations, truncations, infos

    def _get_observation(self, agent: str) -> np.ndarray:
        """
        Get observation for a specific agent.

        Partial observability is controlled by params:
        - observe_price: Include (possibly noisy) price
        - observe_own_inventory: Include (possibly noisy) own inventory
        - observe_own_cumulative: Include own cumulative trades
        - observe_time: Include elapsed time
        - observe_aggregate_volume: Include total market volume
        - observe_num_agents: Include number of agents
        - observe_last_execution_price: Include last execution price
        - observe_last_trade_size: Include last trade size

        Noise can be added via price_noise_std and inventory_noise_std.
        """
        p = self.params
        idx = self.agent_name_to_idx[agent]

        obs_list = []

        # Price (with permanent impact from all agents)
        if p.observe_price:
            price_obs = self.S_impacted
            # Add noise if specified
            if p.price_noise_std > 0:
                noise = np.random.normal(0, p.price_noise_std * p.S_0)
                price_obs = price_obs + noise
            if self.normalize_obs:
                obs_list.append(price_obs / p.S_0)
            else:
                obs_list.append(price_obs)

        # Own inventory
        if p.observe_own_inventory:
            inv_obs = self.inventories[idx]
            # Add noise if specified
            if p.inventory_noise_std > 0:
                noise = np.random.normal(0, p.inventory_noise_std * p.X_0_list[idx])
                inv_obs = max(0, inv_obs + noise)  # Can't be negative
            if self.normalize_obs:
                obs_list.append(inv_obs / p.X_0_list[idx])
            else:
                obs_list.append(inv_obs)

        # Own cumulative trades
        if p.observe_own_cumulative:
            if self.normalize_obs:
                obs_list.append(self.cumulative_traded[idx] / p.X_0_list[idx])
            else:
                obs_list.append(self.cumulative_traded[idx])

        # Time
        if p.observe_time:
            if self.normalize_obs:
                obs_list.append(self.t / p.T)
            else:
                obs_list.append(self.t)

        # Aggregate market volume
        if p.observe_aggregate_volume:
            total_initial = sum(p.X_0_list)
            if self.normalize_obs:
                obs_list.append(self.total_market_volume / total_initial)
            else:
                obs_list.append(self.total_market_volume)

        # Number of agents
        if p.observe_num_agents:
            obs_list.append(float(p.n_agents))

        # Last execution price (for this agent)
        if p.observe_last_execution_price:
            if self.normalize_obs:
                obs_list.append(self.last_execution_prices[idx] / p.S_0)
            else:
                obs_list.append(self.last_execution_prices[idx])

        # Last trade size (for this agent)
        if p.observe_last_trade_size:
            if self.normalize_obs:
                obs_list.append(self.last_trade_sizes[idx] / p.X_0_list[idx])
            else:
                obs_list.append(self.last_trade_sizes[idx])

        # Ensure at least one observation
        if len(obs_list) == 0:
            obs_list.append(0.0)

        return np.array(obs_list, dtype=np.float32)

    def _compute_reward(
        self,
        agent_idx: int,
        shares_traded: float,
        exec_price: float
    ) -> float:
        """Compute reward for an agent."""
        p = self.params

        # Normalization factor
        norm = (p.S_0 * p.X_0_list[agent_idx]) if self.normalize_reward else 1.0

        if self.reward_type == 'pnl':
            reward = shares_traded * exec_price / norm
        elif self.reward_type == 'shortfall':
            shortfall = (self.arrival_price - exec_price) * shares_traded
            reward = -shortfall / norm
        else:
            reward = shares_traded * exec_price / norm

        return float(reward)

    def _get_info(self, agent: str, respect_partial_obs: bool = False) -> Dict[str, Any]:
        """
        Get info dict for an agent.

        Args:
            agent: Agent name
            respect_partial_obs: If True, hide info components that are hidden
                in observation space (prevents info leakage). If False, return
                full info (useful for logging/debugging but NOT for policies).
        """
        p = self.params
        idx = self.agent_name_to_idx[agent]

        # Always provide step/time info (needed for TWAP etc.)
        info = {
            'agent_idx': idx,
            'step': self.step_count,
            'remaining_steps': self.n_steps - self.step_count,
            'n_steps': self.n_steps,
        }

        # Respect partial observability if requested
        if respect_partial_obs:
            if p.observe_price:
                info['price'] = self.S
                info['price_impacted'] = self.S_impacted
            if p.observe_own_inventory:
                info['inventory'] = self.inventories[idx]
                info['initial_inventory'] = p.X_0_list[idx]
            if p.observe_own_cumulative:
                info['cumulative_traded'] = self.cumulative_traded[idx]
                info['cash'] = self.cash[idx]
            if p.observe_time:
                info['time'] = self.t
        else:
            # Full info (for logging/debugging only, NOT for policy decisions)
            info['inventory'] = self.inventories[idx]
            info['initial_inventory'] = p.X_0_list[idx]
            info['cash'] = self.cash[idx]
            info['cumulative_traded'] = self.cumulative_traded[idx]
            info['price'] = self.S
            info['price_impacted'] = self.S_impacted
            info['time'] = self.t

            # Implementation shortfall (only meaningful with full info)
            ideal_value = self.arrival_price * (p.X_0_list[idx] - self.inventories[idx])
            shortfall = ideal_value - self.cash[idx]
            shortfall_bps = 10000 * shortfall / (p.S_0 * p.X_0_list[idx]) if p.X_0_list[idx] > 0 else 0
            info['shortfall'] = shortfall
            info['shortfall_bps'] = shortfall_bps

        return info

    def _get_info_for_policy(self, agent: str) -> Dict[str, Any]:
        """
        Get info dict respecting partial observability.

        Use this when info will be passed to a policy for decision-making.
        Hidden state components are excluded to prevent info leakage.
        """
        return self._get_info(agent, respect_partial_obs=True)

    def _get_info_full(self, agent: str) -> Dict[str, Any]:
        """
        Get full info dict (for logging/debugging).

        WARNING: Do not pass this to policies in partial observability settings,
        as it leaks hidden state information.
        """
        return self._get_info(agent, respect_partial_obs=False)

    def render(self):
        """Render current state."""
        if self.render_mode == 'pygame' and self.renderer is not None:
            p = self.params

            # Collect per-agent data
            agent_data = {}
            for agent in self.agents:
                idx = self.agent_name_to_idx[agent]
                info = self._get_info(agent)

                # Get reward from history if available
                reward = 0.0
                if len(self.history['rewards']) > 0:
                    last_rewards = self.history['rewards'][-1]
                    reward = last_rewards.get(agent, 0.0)

                agent_data[agent] = {
                    'inventory': self.inventories[idx],
                    'shares_traded': self.last_trade_sizes[idx],
                    'execution_price': self.last_execution_prices[idx],
                    'cash': self.cash[idx],
                    'reward': reward,
                    'shortfall_bps': info['shortfall_bps'],
                }

            # Update renderer
            self.renderer.update(
                step=self.step_count,
                t=self.t,
                price=self.S if self.S is not None else p.S_0,
                impacted_price=self.S_impacted if self.S_impacted is not None else p.S_0,
                agent_data=agent_data,
            )

            # Render and return
            return self.renderer.render()

        elif self.render_mode == 'human':
            p = self.params
            print(f"\n{'='*60}")
            print(f"Step {self.step_count}/{self.n_steps} (t={self.t:.3f}/{p.T:.3f})")
            print(f"Price: ${self.S:.2f} → ${self.S_impacted:.2f} (impact: ${self.S - self.S_impacted:.4f})")
            print(f"{'='*60}")

            for agent in self.agents:
                idx = self.agent_name_to_idx[agent]
                info = self._get_info(agent)
                pct_remaining = 100 * self.inventories[idx] / p.X_0_list[idx]
                print(f"  {agent}:")
                print(f"    Inventory: {self.inventories[idx]:,.0f} ({pct_remaining:.1f}% remaining)")
                print(f"    Cash: ${self.cash[idx]:,.2f}")
                print(f"    Shortfall: {info['shortfall_bps']:.1f} bps")

            print(f"  Total market volume: {self.total_market_volume:,.0f}")
            print("-" * 60)

        elif self.render_mode == 'ansi':
            lines = [f"t={self.t:.2f} S=${self.S_impacted:.2f}"]
            for agent in self.agents:
                idx = self.agent_name_to_idx[agent]
                lines.append(f"{agent}: q={self.inventories[idx]:.0f} cash=${self.cash[idx]:.2f}")
            return " | ".join(lines)

    def close(self):
        """Clean up."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def get_history(self) -> Dict[str, list]:
        """Get episode history."""
        return self.history


def raw_env(**kwargs) -> MultiAgentAlmgrenChrissEnv:
    """Create raw environment (PettingZoo convention)."""
    return MultiAgentAlmgrenChrissEnv(**kwargs)


def env(**kwargs):
    """Create wrapped environment with standard wrappers."""
    e = raw_env(**kwargs)
    if wrappers is not None:
        e = wrappers.OrderEnforcingWrapper(e)
    return e


def aec_env(**kwargs):
    """Create AEC (turn-based) version of the environment."""
    if parallel_to_aec is None:
        raise ImportError("pettingzoo is required for AEC conversion")
    return parallel_to_aec(raw_env(**kwargs))


def make_multi_agent_ac_env(
    n_agents: int = 2,
    X_0: float = 500_000,
    sigma: float = 0.02,
    gamma: float = 2.5e-7,
    eta: float = 2.5e-6,
    T: float = 1.0,
    n_steps: int = 20,
    observe_aggregate_volume: bool = False,
    S_0: float = 100.0,
    mu: float = 0.0,
    **kwargs
) -> MultiAgentAlmgrenChrissEnv:
    """
    Create multi-agent Almgren-Chriss environment.

    Args:
        n_agents: Number of trading agents
        X_0: Initial inventory per agent (or list for heterogeneous)
        sigma: Price volatility
        gamma: Permanent impact parameter (shared)
        eta: Temporary impact parameter (or list for heterogeneous)
        T: Trading horizon
        n_steps: Number of trading periods
        observe_aggregate_volume: Include aggregate volume in observations
        S_0: Initial stock price
        mu: Drift per env time unit
        **kwargs: Additional environment arguments

    Returns:
        Configured environment

    Examples:
        >>> # Symmetric 2-agent game
        >>> env = make_multi_agent_ac_env(n_agents=2)

        >>> # Asymmetric: one large, one small trader
        >>> params = MultiAgentACParams(
        ...     n_agents=2,
        ...     X_0_list=[1_000_000, 100_000],  # Large vs small
        ...     eta_list=[2e-6, 5e-6],          # Different costs
        ... )
        >>> env = MultiAgentAlmgrenChrissEnv(params=params)
    """
    params = MultiAgentACParams(
        n_agents=n_agents,
        S_0=S_0,
        X_0=X_0,
        mu=mu,
        sigma=sigma,
        gamma=gamma,
        eta=eta,
        T=T,
        observe_aggregate_volume=observe_aggregate_volume,
    )

    return MultiAgentAlmgrenChrissEnv(
        params=params,
        n_steps=n_steps,
        **kwargs
    )


def make_multi_agent_ac_env_with_ticker(
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
    # Multi-agent AC env params
    n_agents: int = 2,
    S_0: Optional[float] = None,
    X_0: float = 500_000,
    gamma: float = 2.5e-7,
    eta: float = 2.5e-6,
    T: float = 1.0,
    n_steps: int = 20,
    observe_aggregate_volume: bool = False,
    return_calibration: bool = False,
    **kwargs,
):
    """
    Build multi-agent AC env by calibrating dynamics from a market ticker.

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

    env = make_multi_agent_ac_env(
        n_agents=n_agents,
        S_0=float(calibration.spot_price if S_0 is None else S_0),
        X_0=X_0,
        mu=calibration.mu_env,
        sigma=calibration.sigma_env,
        gamma=gamma,
        eta=eta,
        T=T,
        n_steps=n_steps,
        observe_aggregate_volume=observe_aggregate_volume,
        **kwargs,
    )
    env.calibration_metadata = calibration

    if return_calibration:
        return env, calibration
    return env


# Local alias for convenience inside this module namespace.
make_env_with_ticker = make_multi_agent_ac_env_with_ticker
