"""Autoencoder + LSTM information state."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .base import InfoState
from .grid_autoencoder.model import GridAutoencoder


class AutoencoderLSTMInfoState(InfoState):
    """Grid autoencoder encoder + LSTM to summarize history."""

    def __init__(
        self,
        obs_shape: Tuple[int, int, int],
        action_dim: int,
        latent_dim: int = 64,
        state_dim: Optional[int] = None,
        num_layers: int = 1,
        pretrained_path: Optional[str] = None,
        freeze_autoencoder: bool = False,
    ):
        if state_dim is None:
            state_dim = latent_dim
        super().__init__(obs_dim=latent_dim, action_dim=action_dim, state_dim=state_dim)
        self.obs_shape = obs_shape
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        self.autoencoder = GridAutoencoder(obs_shape, latent_dim=latent_dim)
        if pretrained_path:
            state = torch.load(pretrained_path, map_location="cpu")
            self.autoencoder.load_state_dict(state, strict=False)
        if freeze_autoencoder:
            for param in self.autoencoder.parameters():
                param.requires_grad = False

        lstm_input_dim = latent_dim + action_dim + 1
        self.lstm = nn.LSTM(lstm_input_dim, state_dim, num_layers=num_layers, batch_first=True)

    def forward(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        reward: Optional[torch.Tensor] = None,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size = obs.shape[0]
        device = obs.device

        is_sequence = obs.dim() == 5
        if not is_sequence:
            obs = obs.unsqueeze(1)
            if action is not None:
                action = action.unsqueeze(1)
            if reward is not None:
                reward = reward.unsqueeze(1)

        seq_len = obs.shape[1]
        if action is None:
            action = torch.zeros(batch_size, seq_len, self.action_dim, device=device)
        if reward is None:
            reward = torch.zeros(batch_size, seq_len, 1, device=device)
        if hidden is None:
            hidden = self.init_hidden(batch_size, device)

        obs_flat = obs.reshape(batch_size * seq_len, *self.obs_shape)
        latent = self.autoencoder.encode(obs_flat)
        latent = latent.reshape(batch_size, seq_len, self.latent_dim)

        x = torch.cat([latent, action, reward], dim=-1)
        out, hidden = self.lstm(x, hidden)
        info_state = out[:, -1, :]
        return info_state, hidden

    def init_hidden(self, batch_size: int = 1, device: str = "cpu"):
        h_0 = torch.zeros(self.num_layers, batch_size, self.state_dim, device=device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.state_dim, device=device)
        return (h_0, c_0)
