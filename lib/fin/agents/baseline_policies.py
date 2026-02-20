"""
Baseline Execution Policies for Almgren-Chriss Environments.

Provides agent classes that can be used as:
- Opponents during training
- Baselines for evaluation
- Components in population-based training

All agents implement a common interface:
    action = agent.act(observation, info)

Author: Yunian Pan
Email: yp1170@nyu.edu
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union
import numpy as np


class BaseAgent(ABC):
    """Abstract base class for execution agents."""

    @abstractmethod
    def act(self, observation: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        """
        Select action given observation and info.

        Args:
            observation: Environment observation
            info: Additional info dict from environment

        Returns:
            Action array (fraction of inventory to trade)
        """
        pass

    def reset(self) -> None:
        """Reset agent state (if any)."""
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__


class TWAPAgent(BaseAgent):
    """
    Time-Weighted Average Price (TWAP) strategy.

    Trades equal fraction of remaining inventory each step.
    Optimal when there's no urgency (lambda=0) and linear impact.
    """

    def act(self, observation: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        remaining_steps = info.get('remaining_steps', 1)
        fraction = 1.0 / max(remaining_steps, 1)
        return np.array([fraction], dtype=np.float32)


class VWAPAgent(BaseAgent):
    """
    Volume-Weighted Average Price (VWAP) strategy.

    Trades proportionally to expected market volume profile.
    Volume profile specifies fraction of INITIAL inventory to trade at each step.
    Profile should sum to 1.0 for full liquidation.

    If None, assumes uniform volume (equivalent to TWAP).
    """

    def __init__(self, volume_profile: Optional[np.ndarray] = None):
        """
        Args:
            volume_profile: Fraction of INITIAL inventory to trade at each step.
                           Should sum to 1.0. If None, uses TWAP.
        """
        self.volume_profile = volume_profile
        self.step = 0
        self.initial_inventory = None
        self.cumulative_traded_frac = 0.0

    def act(self, observation: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        if self.volume_profile is None:
            # Uniform volume = TWAP
            remaining_steps = info.get('remaining_steps', 1)
            fraction = 1.0 / max(remaining_steps, 1)
        else:
            # Track initial inventory on first call
            if self.initial_inventory is None:
                self.initial_inventory = info.get('initial_inventory', info.get('inventory', 1.0))

            # Get target fraction of INITIAL inventory for this step
            if self.step < len(self.volume_profile):
                target_frac_of_initial = self.volume_profile[self.step]
            else:
                # Liquidate remaining
                target_frac_of_initial = 1.0 - self.cumulative_traded_frac

            # Convert to fraction of CURRENT inventory
            current_inventory = info.get('inventory', 1.0)
            if current_inventory > 0 and self.initial_inventory > 0:
                shares_to_trade = target_frac_of_initial * self.initial_inventory
                fraction = min(shares_to_trade / current_inventory, 1.0)
            else:
                fraction = 1.0

            self.cumulative_traded_frac += target_frac_of_initial

        self.step += 1
        return np.array([fraction], dtype=np.float32)

    def reset(self) -> None:
        self.step = 0
        self.initial_inventory = None
        self.cumulative_traded_frac = 0.0


class OptimalACAgent(BaseAgent):
    """
    Optimal Almgren-Chriss execution strategy.

    Implements the closed-form risk-averse optimal strategy:
        x*(t) = X_0 * sinh(kappa * (T - t)) / sinh(kappa * T)

    where kappa = sqrt(lambda * sigma^2 / eta)

    This is optimal for:
    - Linear permanent and temporary impact
    - CARA utility with risk aversion lambda
    - Known initial inventory X_0
    """

    def __init__(
        self,
        kappa: float,
        T: float,
        n_steps: int,
        X_0: Optional[float] = None,
    ):
        """
        Args:
            kappa: Risk-adjustment parameter sqrt(lambda * sigma^2 / eta)
            T: Total time horizon
            n_steps: Number of trading steps
            X_0: Initial inventory (if None, inferred from info)
        """
        self.kappa = kappa
        self.T = T
        self.n_steps = n_steps
        self.dt = T / n_steps
        self.X_0 = X_0
        self.step = 0

    @classmethod
    def from_env(cls, env) -> 'OptimalACAgent':
        """Create agent from environment parameters."""
        return cls(
            kappa=env.kappa,
            T=env.params.T,
            n_steps=env.n_steps,
            X_0=env.params.X_0,
        )

    @classmethod
    def from_params(
        cls,
        sigma: float,
        eta: float,
        lambda_var: float,
        T: float,
        n_steps: int,
        X_0: Optional[float] = None,
    ) -> 'OptimalACAgent':
        """Create agent from market parameters."""
        kappa = np.sqrt(lambda_var * sigma**2 / eta) if eta > 0 else 0.0
        return cls(kappa=kappa, T=T, n_steps=n_steps, X_0=X_0)

    def optimal_inventory(self, t: float) -> float:
        """Optimal inventory at time t."""
        if self.X_0 is None:
            raise ValueError("X_0 must be set to compute optimal inventory")

        if self.kappa < 1e-10:
            # Risk-neutral: linear liquidation (TWAP)
            return self.X_0 * (1 - t / self.T)

        return self.X_0 * np.sinh(self.kappa * (self.T - t)) / np.sinh(self.kappa * self.T)

    def optimal_trade_rate(self, t: float) -> float:
        """Optimal shares to trade per unit time at time t."""
        if self.X_0 is None:
            raise ValueError("X_0 must be set to compute optimal trade rate")

        if self.kappa < 1e-10:
            return self.X_0 / self.T

        return (self.X_0 * self.kappa * np.cosh(self.kappa * (self.T - t)) /
                np.sinh(self.kappa * self.T))

    def act(self, observation: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        # Get current state
        t = self.step * self.dt
        current_inventory = info.get('inventory', self.X_0)

        if self.X_0 is None:
            self.X_0 = info.get('initial_inventory', current_inventory)

        if current_inventory <= 0:
            self.step += 1
            return np.array([0.0], dtype=np.float32)

        # Compute optimal shares to trade this period
        optimal_shares = self.optimal_trade_rate(t) * self.dt
        fraction = min(optimal_shares / current_inventory, 1.0)

        self.step += 1
        return np.array([fraction], dtype=np.float32)

    def reset(self) -> None:
        self.step = 0


class AggressiveAgent(BaseAgent):
    """
    Aggressive (front-loaded) execution strategy.

    Trades a fixed high fraction early, then slows down.
    Minimizes timing risk but incurs higher temporary impact.
    """

    def __init__(self, early_fraction: float = 0.4, late_fraction: float = 0.15,
                 switch_step: int = 3):
        """
        Args:
            early_fraction: Fraction to trade in early steps
            late_fraction: Fraction to trade in late steps
            switch_step: Step at which to switch from early to late
        """
        self.early_fraction = early_fraction
        self.late_fraction = late_fraction
        self.switch_step = switch_step
        self.step = 0

    def act(self, observation: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        if self.step < self.switch_step:
            fraction = self.early_fraction
        else:
            fraction = self.late_fraction

        self.step += 1
        return np.array([min(fraction, 1.0)], dtype=np.float32)

    def reset(self) -> None:
        self.step = 0


class ConservativeAgent(BaseAgent):
    """
    Conservative (back-loaded) execution strategy.

    Trades slowly early, accelerates toward deadline.
    Lower temporary impact but higher timing risk.
    """

    def __init__(self, base_fraction: float = 0.08, acceleration: float = 1.3):
        """
        Args:
            base_fraction: Base fraction to trade
            acceleration: Multiplicative increase per step
        """
        self.base_fraction = base_fraction
        self.acceleration = acceleration
        self.step = 0

    def act(self, observation: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        fraction = self.base_fraction * (self.acceleration ** self.step)
        self.step += 1
        return np.array([min(fraction, 1.0)], dtype=np.float32)

    def reset(self) -> None:
        self.step = 0


class RandomAgent(BaseAgent):
    """
    Random execution strategy.

    Trades random fraction each step. Used as baseline.
    """

    def __init__(self, min_fraction: float = 0.0, max_fraction: float = 0.3,
                 seed: Optional[int] = None):
        """
        Args:
            min_fraction: Minimum fraction to trade
            max_fraction: Maximum fraction to trade
            seed: Random seed for reproducibility
        """
        self.min_fraction = min_fraction
        self.max_fraction = max_fraction
        self.rng = np.random.default_rng(seed)

    def act(self, observation: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        fraction = self.rng.uniform(self.min_fraction, self.max_fraction)
        return np.array([fraction], dtype=np.float32)

    def reset(self) -> None:
        pass  # RNG state persists


class ConstantFractionAgent(BaseAgent):
    """
    Constant fraction execution strategy.

    Trades fixed fraction each step regardless of state.
    """

    def __init__(self, fraction: float = 0.1):
        self.fraction = fraction

    def act(self, observation: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        return np.array([self.fraction], dtype=np.float32)


class AdaptiveAgent(BaseAgent):
    """
    Adaptive execution strategy that adjusts based on price movements.

    Trades more when price is favorable, less when unfavorable.
    Simple momentum-based heuristic.
    """

    def __init__(self, base_fraction: float = 0.1, sensitivity: float = 2.0,
                 initial_price: Optional[float] = None):
        """
        Args:
            base_fraction: Base fraction to trade
            sensitivity: How much to adjust based on price
            initial_price: Reference price (if None, uses first observed)
        """
        self.base_fraction = base_fraction
        self.sensitivity = sensitivity
        self.initial_price = initial_price
        self.last_price = None

    def act(self, observation: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        # Assume price is first element of observation (normalized)
        current_price = observation[0] if len(observation) > 0 else 1.0

        if self.last_price is None:
            self.last_price = current_price
            return np.array([self.base_fraction], dtype=np.float32)

        # Price dropped -> trade more (selling into weakness)
        # Price rose -> trade less (wait for better prices)
        price_change = (current_price - self.last_price) / self.last_price
        adjustment = 1.0 - self.sensitivity * price_change
        adjustment = np.clip(adjustment, 0.5, 2.0)

        fraction = self.base_fraction * adjustment
        self.last_price = current_price

        return np.array([min(fraction, 1.0)], dtype=np.float32)

    def reset(self) -> None:
        self.last_price = None


# Multi-agent wrapper
class MultiAgentPolicy:
    """
    Wrapper to use single-agent policies in multi-agent environments.

    Assigns a policy to each agent.
    """

    def __init__(self, policies: Dict[str, BaseAgent]):
        """
        Args:
            policies: Dict mapping agent name to policy
        """
        self.policies = policies

    def act(self, observations: Dict[str, np.ndarray],
            infos: Dict[str, Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Get actions for all agents."""
        actions = {}
        for agent_name, obs in observations.items():
            if agent_name in self.policies:
                info = infos.get(agent_name, {})
                actions[agent_name] = self.policies[agent_name].act(obs, info)
        return actions

    def reset(self) -> None:
        for policy in self.policies.values():
            policy.reset()

    @classmethod
    def all_twap(cls, agent_names: list) -> 'MultiAgentPolicy':
        """Create policy with TWAP for all agents."""
        return cls({name: TWAPAgent() for name in agent_names})

    @classmethod
    def all_same(cls, agent_names: list, policy: BaseAgent) -> 'MultiAgentPolicy':
        """Create policy with same strategy for all agents."""
        # Note: shares same instance, be careful with stateful policies
        return cls({name: policy for name in agent_names})


# Utility function for environment method integration
def get_optimal_agent_for_env(env) -> OptimalACAgent:
    """Create optimal agent configured for given environment."""
    if hasattr(env, 'kappa'):
        return OptimalACAgent.from_env(env)
    else:
        # Multi-agent env - use shared params
        params = env.params
        return OptimalACAgent.from_params(
            sigma=params.sigma,
            eta=params.eta if hasattr(params, 'eta') else params.eta_list[0],
            lambda_var=params.lambda_var,
            T=params.T,
            n_steps=env.n_steps,
            X_0=params.X_0 if hasattr(params, 'X_0') else params.X_0_list[0],
        )
