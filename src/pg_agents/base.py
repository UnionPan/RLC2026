"""Base agent class for policy gradient methods."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn


class Agent(ABC, nn.Module):
    """Abstract base class for RL agents."""

    def __init__(self, info_state_encoder, action_dim: int, config: Dict[str, Any]):
        super().__init__()
        self.info_state_encoder = info_state_encoder
        self.action_dim = action_dim
        self.config = config
        self.device = config.get('device', 'cpu')

    @abstractmethod
    def get_action(
        self,
        obs: torch.Tensor,
        prev_action: Any = None,
        prev_reward: Any = None,
        hidden: Any = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """Select action given observation.

        Args:
            obs: Observation tensor
            hidden: Hidden state for recurrent info state
            deterministic: If True, select argmax action

        Returns:
            action: Selected action (discrete index or continuous value)
            log_prob: Log probability of action
            hidden: Updated hidden state
        """
        pass

    @abstractmethod
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        hidden: Any = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for training.

        Args:
            obs: Batch of observations (batch, obs_dim) or (batch, seq_len, obs_dim)
            actions: Batch of actions (batch,) or (batch, seq_len)
            hidden: Hidden states

        Returns:
            log_probs: Log probabilities of actions (batch,)
            values: Value estimates (batch,) if using value function
            entropy: Entropy of policy (batch,)
        """
        pass

    @abstractmethod
    def update(self, rollout_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update agent from rollout data.

        Args:
            rollout_data: Dictionary containing:
                - obs: Observations
                - actions: Actions taken
                - rewards: Rewards received
                - dones: Episode termination flags
                - values: Value estimates (if applicable)
                - log_probs: Log probabilities

        Returns:
            Dictionary of training metrics
        """
        pass

    def to_device(self, device: str):
        """Move agent to device."""
        self.device = device
        self.to(device)
        return self
