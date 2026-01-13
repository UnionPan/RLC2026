"""Autoencoder + temporal Transformer information state."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .base import InfoState
from .grid_autoencoder.model import GridAutoencoder


class AutoencoderTransformerInfoState(InfoState):
    """Grid autoencoder encoder + temporal Transformer to summarize history."""

    def __init__(
        self,
        obs_shape,
        action_dim: int,
        latent_dim: int = 64,
        model_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 64,
        pretrained_path: Optional[str] = None,
        freeze_autoencoder: bool = False,
    ):
        super().__init__(obs_dim=model_dim, action_dim=action_dim, state_dim=model_dim)
        self.obs_shape = obs_shape
        self.latent_dim = latent_dim
        self.model_dim = model_dim
        self.max_seq_len = max_seq_len

        self.autoencoder = GridAutoencoder(obs_shape, latent_dim=latent_dim)
        if pretrained_path:
            state = torch.load(pretrained_path, map_location="cpu")
            self.autoencoder.load_state_dict(state, strict=False)
        if freeze_autoencoder:
            for param in self.autoencoder.parameters():
                param.requires_grad = False

        token_dim = latent_dim + action_dim + 1
        self.input_proj = nn.Linear(token_dim, model_dim)
        self.pos_embed = nn.Embedding(max_seq_len, model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        reward: Optional[torch.Tensor] = None,
        hidden: Optional[torch.Tensor] = None,
    ):
        device = obs.device

        if obs.dim() == 5:
            return self._forward_sequence(obs, action, reward)
        return self._forward_step(obs, action, reward, hidden)

    def _forward_sequence(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor],
        reward: Optional[torch.Tensor],
    ):
        batch_size, seq_len = obs.shape[:2]
        device = obs.device

        if action is None:
            action = torch.zeros(batch_size, seq_len, self.action_dim, device=device)
        if reward is None:
            reward = torch.zeros(batch_size, seq_len, 1, device=device)

        obs_flat = obs.reshape(batch_size * seq_len, *self.obs_shape)
        latent = self.autoencoder.encode(obs_flat).reshape(batch_size, seq_len, self.latent_dim)
        tokens = torch.cat([latent, action, reward], dim=-1)
        tokens = self.input_proj(tokens)

        tokens = tokens + self._positional_encoding(seq_len, device)
        attn_mask = self._causal_mask(seq_len, device)
        encoded = self.transformer(tokens, mask=attn_mask)
        info_state = encoded[:, -1, :]
        return info_state, encoded

    def _forward_step(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor],
        reward: Optional[torch.Tensor],
        hidden: Optional[torch.Tensor],
    ):
        batch_size = obs.shape[0]
        device = obs.device

        if action is None:
            action = torch.zeros(batch_size, self.action_dim, device=device)
        if reward is None:
            reward = torch.zeros(batch_size, 1, device=device)

        latent = self.autoencoder.encode(obs)
        token = torch.cat([latent, action, reward], dim=-1)
        token = self.input_proj(token).unsqueeze(1)

        if hidden is None:
            seq = token
        else:
            seq = torch.cat([hidden, token], dim=1)
            if seq.shape[1] > self.max_seq_len:
                seq = seq[:, -self.max_seq_len :, :]

        seq_len = seq.shape[1]
        seq = seq + self._positional_encoding(seq_len, device)
        attn_mask = self._causal_mask(seq_len, device)
        encoded = self.transformer(seq, mask=attn_mask)
        info_state = encoded[:, -1, :]
        return info_state, encoded

    def init_hidden(self, batch_size: int = 1, device: str = "cpu"):
        return None

    def _positional_encoding(self, seq_len: int, device: torch.device):
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        return self.pos_embed(positions)

    def _causal_mask(self, seq_len: int, device: torch.device):
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
