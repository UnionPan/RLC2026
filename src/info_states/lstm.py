"""LSTM-based information state encoder."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import numpy as np

from .base import InfoState


class LSTMInfoState(InfoState):
    """LSTM-based information state encoder (similar to AIS rho network)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        state_dim: int = 64,
        embed_dim: int = 128,
        num_layers: int = 1,
    ):
        super().__init__(obs_dim, action_dim, state_dim)
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # Input: obs + action + reward
        input_dim = obs_dim + action_dim + 1

        # Embedding layer
        self.embed = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU()
        )

        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            embed_dim,
            state_dim,
            num_layers=num_layers,
            batch_first=True
        )

    def forward(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        reward: Optional[torch.Tensor] = None,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            obs: (batch, obs_dim) or (batch, seq_len, obs_dim)
            action: (batch, action_dim) or (batch, seq_len, action_dim)
            reward: (batch, 1) or (batch, seq_len, 1)
            hidden: (h_0, c_0) each (num_layers, batch, state_dim)

        Returns:
            info_state: (batch, state_dim) - last timestep output
            hidden: Updated (h_n, c_n)
        """
        batch_size = obs.shape[0]
        device = obs.device

        # Handle default inputs
        if action is None:
            if len(obs.shape) == 2:
                action = torch.zeros(batch_size, self.action_dim, device=device)
            else:
                seq_len = obs.shape[1]
                action = torch.zeros(batch_size, seq_len, self.action_dim, device=device)

        if reward is None:
            if len(obs.shape) == 2:
                reward = torch.zeros(batch_size, 1, device=device)
            else:
                seq_len = obs.shape[1]
                reward = torch.zeros(batch_size, seq_len, 1, device=device)

        if hidden is None:
            hidden = self.init_hidden(batch_size, device)

        # Concatenate inputs
        if len(obs.shape) == 2:
            # Single timestep: (batch, obs_dim)
            x = torch.cat([obs, action, reward], dim=-1)
            x = x.unsqueeze(1)  # (batch, 1, input_dim)
        else:
            # Sequence: (batch, seq_len, obs_dim)
            x = torch.cat([obs, action, reward], dim=-1)

        # Embed and process through LSTM
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)

        # Return last timestep as info state
        info_state = out[:, -1, :]  # (batch, state_dim)

        return info_state, hidden

    def init_hidden(self, batch_size: int = 1, device: str = 'cpu'):
        """Initialize LSTM hidden state."""
        h_0 = torch.zeros(self.num_layers, batch_size, self.state_dim, device=device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.state_dim, device=device)
        return (h_0, c_0)


class CNNLSTMInfoState(InfoState):
    """CNN + LSTM for grid-based observations (e.g., competitive_envs)."""

    def __init__(
        self,
        obs_shape: Tuple[int, int, int],  # (H, W, C) e.g., (5, 5, 7)
        action_dim: int,
        state_dim: int = 64,
        cnn_channels: list = [16, 32],
    ):
        # Flatten obs for base class
        obs_dim = int(np.prod(obs_shape))
        super().__init__(obs_dim, action_dim, state_dim)

        self.obs_shape = obs_shape
        self.cnn_channels = cnn_channels

        # CNN encoder for spatial observations
        H, W, C = obs_shape
        layers = []
        in_channels = C
        for out_channels in cnn_channels:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
            ])
            in_channels = out_channels

        self.cnn = nn.Sequential(*layers)

        # Calculate CNN output size
        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            cnn_out = self.cnn(dummy)
            cnn_out_dim = cnn_out.numel()

        # Input to LSTM: CNN features + action + reward
        lstm_input_dim = cnn_out_dim + action_dim + 1

        self.embed = nn.Sequential(
            nn.Linear(lstm_input_dim, state_dim),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(state_dim, state_dim, batch_first=True)

    def forward(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        reward: Optional[torch.Tensor] = None,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            obs: (batch, H, W, C) or (batch, seq_len, H, W, C)
            action: (batch, action_dim) or (batch, seq_len, action_dim)
            reward: (batch, 1) or (batch, seq_len, 1)
            hidden: LSTM hidden state

        Returns:
            info_state: (batch, state_dim)
            hidden: Updated hidden state
        """
        batch_size = obs.shape[0]
        device = obs.device

        # Handle sequence vs single timestep
        is_sequence = len(obs.shape) == 5
        if not is_sequence:
            obs = obs.unsqueeze(1)  # (batch, 1, H, W, C)
            if action is not None:
                action = action.unsqueeze(1)
            if reward is not None:
                reward = reward.unsqueeze(1)

        seq_len = obs.shape[1]

        # Default inputs
        if action is None:
            action = torch.zeros(batch_size, seq_len, self.action_dim, device=device)
        if reward is None:
            reward = torch.zeros(batch_size, seq_len, 1, device=device)
        if hidden is None:
            hidden = self.init_hidden(batch_size, device)

        # Process each timestep through CNN
        H, W, C = self.obs_shape
        obs_flat = obs.reshape(batch_size * seq_len, H, W, C)
        obs_flat = obs_flat.permute(0, 3, 1, 2)  # (B*T, C, H, W)

        cnn_out = self.cnn(obs_flat)  # (B*T, channels, H, W)
        cnn_out = cnn_out.reshape(batch_size, seq_len, -1)  # (B, T, features)

        # Concatenate with action and reward
        x = torch.cat([cnn_out, action, reward], dim=-1)

        # Embed and process through LSTM
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)

        # Return last timestep
        info_state = out[:, -1, :]

        return info_state, hidden

    def init_hidden(self, batch_size: int = 1, device: str = 'cpu'):
        h_0 = torch.zeros(1, batch_size, self.state_dim, device=device)
        c_0 = torch.zeros(1, batch_size, self.state_dim, device=device)
        return (h_0, c_0)