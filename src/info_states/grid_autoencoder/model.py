"""CNN autoencoder for grid observations."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class GridAutoencoder(nn.Module):
    """Lightweight CNN autoencoder for grid-based observations."""

    def __init__(self, obs_shape: Tuple[int, int, int], latent_dim: int = 64):
        super().__init__()
        self.obs_shape = obs_shape
        self.latent_dim = latent_dim

        height, width, channels = obs_shape
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, channels, height, width)
            conv_out = self.encoder_conv(dummy)
        self._conv_shape = conv_out.shape[1:]
        conv_dim = int(conv_out.numel())

        self.encoder_fc = nn.Linear(conv_dim, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, conv_dim)

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_nchw(x)
        features = self.encoder_conv(x)
        flat = features.reshape(features.shape[0], -1)
        return self.encoder_fc(flat)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        flat = self.decoder_fc(z)
        features = flat.reshape(z.shape[0], *self._conv_shape)
        recon = self.decoder_conv(features)
        return self._to_nhwc(recon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)

    def _to_nchw(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        return x.permute(0, 3, 1, 2)

    def _to_nhwc(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 3, 1)
