"""Train a grid autoencoder on offline rollouts."""

from __future__ import annotations

import argparse
import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

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
    model_dir: str,
    log_dir: str,
):
    dataset = GridRolloutDataset(observations)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = GridAutoencoder(obs_shape, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    os.makedirs(model_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(1, epochs + 1):
        losses = []
        global_step = (epoch - 1) * len(loader)
        for batch in loader:
            batch = batch.to(device)
            recon = model(batch)
            loss = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            writer.add_scalar("train/batch_loss", loss.item(), global_step)
            global_step += 1

        avg_loss = float(np.mean(losses)) if losses else 0.0
        print(f"Epoch {epoch}/{epochs} loss={avg_loss:.6f}")
        writer.add_scalar("train/epoch_loss", avg_loss, epoch)
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(model_dir, f"autoencoder_ep{epoch}.pth"))

    torch.save(model.state_dict(), os.path.join(model_dir, "autoencoder_final.pth"))
    writer.close()


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
    parser.add_argument("--log_dir", default=None)
    args = parser.parse_args()

    run_id = f"{args.env_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(args.output_dir, run_id)
    data_dir = os.path.join(run_dir, "data")
    model_dir = os.path.join(run_dir, "models")
    log_dir = args.log_dir or os.path.join(run_dir, "tb")

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
        os.makedirs(data_dir, exist_ok=True)
        dataset_path = os.path.join(data_dir, "rollouts.npy")
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
        model_dir=model_dir,
        log_dir=log_dir,
    )


if __name__ == "__main__":
    main()
