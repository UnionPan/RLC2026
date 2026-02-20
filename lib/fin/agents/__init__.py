"""
Execution Agents for RL Environments.

Provides baseline policies and agent wrappers for:
- Single-agent Almgren-Chriss execution
- Multi-agent POSG execution
- Nash equilibrium benchmarks for multi-agent games

Author: Yunian Pan
Email: yp1170@nyu.edu
"""

from .baseline_policies import (
    BaseAgent,
    TWAPAgent,
    VWAPAgent,
    OptimalACAgent,
    AggressiveAgent,
    ConservativeAgent,
    RandomAgent,
    ConstantFractionAgent,
    AdaptiveAgent,
    MultiAgentPolicy,
    get_optimal_agent_for_env,
)

from .nash_equilibrium import (
    NashEquilibriumParams,
    SymmetricNashAgent,
    CooperativeAgent,
    AsymmetricNashSolver,
    AsymmetricNashAgent,
    compute_price_of_anarchy,
    create_nash_multi_agent_policy,
    create_cooperative_multi_agent_policy,
)

__all__ = [
    # Baseline policies
    'BaseAgent',
    'TWAPAgent',
    'VWAPAgent',
    'OptimalACAgent',
    'AggressiveAgent',
    'ConservativeAgent',
    'RandomAgent',
    'ConstantFractionAgent',
    'AdaptiveAgent',
    'MultiAgentPolicy',
    'get_optimal_agent_for_env',
    # Nash equilibrium
    'NashEquilibriumParams',
    'SymmetricNashAgent',
    'CooperativeAgent',
    'AsymmetricNashSolver',
    'AsymmetricNashAgent',
    'compute_price_of_anarchy',
    'create_nash_multi_agent_policy',
    'create_cooperative_multi_agent_policy',
]
