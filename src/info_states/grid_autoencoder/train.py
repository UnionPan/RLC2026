"""Train a grid autoencoder on offline rollouts."""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import collect_rollouts, load_rollouts, save_rollouts, GridRolloutDataset
from .model import GridAutoencoder


def train(
    observations: np.ndarray,
    obs_shape: tuple,
    latent_dim: int,
    batch_size: int,
    epochs: int,
    lr: float,
    device: str,
    output_dir: str,
):
    dataset = GridRolloutDataset(observations)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = GridAutoencoder(obs_shape, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        losses = []
        for batch in loader:
            batch = batch.to(device)
            recon = model(batch)
            loss = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        avg_loss = float(np.mean(losses)) if losses else 0.0
        print(f"Epoch {epoch}/{epochs} loss={avg_loss:.6f}")
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(output_dir, f"autoencoder_ep{epoch}.pth"))

    torch.save(model.state_dict(), os.path.join(output_dir, "autoencoder_final.pth"))


def main():
    parser = argparse.ArgumentParser(description="Grid autoencoder training")
    parser.add_argument("--env_name", default="competitive_fourrooms")
    parser.add_argument("--output_dir", default="grid_autoencoder_runs")
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--collect_only", action="store_true")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=11)
    parser.add_argument("--height", type=int, default=11)
    parser.add_argument("--view_radius", type=int, default=2)
    parser.add_argument("--wall_density", type=float, default=0.1)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.dataset_path:
        observations = load_rollouts(args.dataset_path)
        obs_shape = observations.shape[1:]
    else:
        observations, obs_shape = collect_rollouts(
            env_name=args.env_name,
            episodes=args.episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            width=args.width,
            height=args.height,
            view_radius=args.view_radius,
            wall_density=args.wall_density,
        )
        dataset_path = os.path.join(args.output_dir, f"{args.env_name}_rollouts.npy")
        save_rollouts(dataset_path, observations)

    if args.collect_only:
        print("Rollout collection completed.")
        return

    train(
        observations=observations,
        obs_shape=obs_shape,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
