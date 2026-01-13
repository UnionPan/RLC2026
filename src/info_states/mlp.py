"""Simple MLP-based information state (non-recurrent baseline)."""

from typing import Optional, Tuple
import torch
import torch.nn as nn

from .base import InfoState


class MLPInfoState(InfoState):
    """Simple MLP baseline - no recurrence, just current observation."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        state_dim: int = 64,
        hidden_dims: list = [128, 128],
    ):
        super().__init__(obs_dim, action_dim, state_dim)

        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, state_dim))

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        reward: Optional[torch.Tensor] = None,
        hidden: Optional[None] = None,
    ) -> Tuple[torch.Tensor, None]:
        """Process only current observation (stateless).

        Args:
            obs: (batch, obs_dim)
            action: Ignored
            reward: Ignored
            hidden: Ignored

        Returns:
            info_state: (batch, state_dim)
            hidden: None
        """
        return self.net(obs), None

    def init_hidden(self, batch_size: int = 1, device: str = 'cpu'):
        return None