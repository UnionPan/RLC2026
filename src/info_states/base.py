"""Base classes for information state representations."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple
import torch
import torch.nn as nn


class InfoState(ABC, nn.Module):
    """Abstract base class for information state encoders."""

    def __init__(self, obs_dim: int, action_dim: int, state_dim: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim

    @abstractmethod
    def forward(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        reward: Optional[torch.Tensor] = None,
        hidden: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, Any]:
        """Compute information state from observation history.

        Args:
            obs: (batch, obs_dim)
            action: (batch, action_dim) - previous action
            reward: (batch, 1) - previous reward
            hidden: Hidden state from previous timestep

        Returns:
            info_state: (batch, state_dim)
            hidden: Updated hidden state
        """
        pass

    @abstractmethod
    def init_hidden(self, batch_size: int = 1, device: str = 'cpu') -> Any:
        """Initialize hidden state."""
        pass