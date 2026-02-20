#!/usr/bin/env python
"""
Train MAPPO agents on Multi-Agent Almgren-Chriss Optimal Execution.

Usage:
    python scripts/train_ac_mappo.py --n_agents 2 --n_episodes 5000
    python scripts/train_ac_mappo.py --n_agents 3 --device cuda --n_episodes 10000

Author: Yunian Pan
Email: yp1170@nyu.edu
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np

from src.training import ACMAPPOTrainer, ACMAPPOConfig


def main(args):
    """Main training function."""
    # Set device with automatic fallback
    device = args.device
    if device == 'cuda':
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"\n✓ GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            print("\n⚠ CUDA not available, falling back to CPU")
            device = 'cpu'
    elif device == 'mps':
        if torch.backends.mps.is_available():
            print("\n✓ MPS (Apple Silicon) detected")
        else:
            print("\n⚠ MPS not available, falling back to CPU")
            device = 'cpu'
    else:
        print(f"\n• Using device: {device}")

    # Create config
    config = ACMAPPOConfig(
        # Environment
        n_agents=args.n_agents,
        n_steps=args.n_steps,
        X_0=args.X_0,
        gamma_impact=args.gamma_impact,
        eta=args.eta,
        sigma=args.sigma,
        lambda_var=args.lambda_var,

        # Action discretization
        n_action_bins=args.n_action_bins,

        # Info state
        info_state_type=args.info_state,
        info_state_dim=args.info_state_dim,
        lstm_embed_dim=args.lstm_embed_dim,
        lstm_num_layers=args.lstm_num_layers,

        # MAPPO
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        n_ppo_epochs=args.n_ppo_epochs,
        minibatch_size=args.minibatch_size,

        # Network
        hidden_dim=args.hidden_dim,
        share_policy=args.share_policy,
        share_value=args.share_value,
        centralized_critic=args.centralized_critic,

        # Training
        n_episodes=args.n_episodes,
        update_freq=args.update_freq,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        save_path=args.save_path,
        device=device,
        seed=args.seed,
    )

    # Print configuration
    print("=" * 60)
    print("Multi-Agent Almgren-Chriss MAPPO Training")
    print("=" * 60)
    print(f"Agents: {config.n_agents}")
    print(f"Steps: {config.n_steps}")
    print(f"Episodes: {config.n_episodes}")
    print(f"Info State: {config.info_state_type} (dim={config.info_state_dim})")
    print(f"Actions: {config.n_action_bins} discrete bins")
    print(f"Device: {device}")
    print(f"Save path: {config.save_path}")
    print("=" * 60)

    # Create trainer
    trainer = ACMAPPOTrainer(config)

    # Load checkpoint if specified
    if args.load_checkpoint:
        trainer.load_checkpoint(args.load_checkpoint)

    # Train
    episode_rewards, training_metrics = trainer.train(progress_bar=not args.no_progress)

    # Save final checkpoint
    trainer.save_checkpoint(config.n_episodes)

    print("\nTraining completed!")

    return trainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train MAPPO on Multi-Agent Almgren-Chriss Execution',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Environment
    env_group = parser.add_argument_group('Environment')
    env_group.add_argument('--n_agents', type=int, default=2,
                          help='Number of trading agents')
    env_group.add_argument('--n_steps', type=int, default=20,
                          help='Number of execution steps')
    env_group.add_argument('--X_0', type=float, default=100_000,
                          help='Initial inventory per agent')
    env_group.add_argument('--gamma_impact', type=float, default=2.5e-7,
                          help='Permanent price impact')
    env_group.add_argument('--eta', type=float, default=2.5e-6,
                          help='Temporary price impact')
    env_group.add_argument('--sigma', type=float, default=0.02,
                          help='Price volatility')
    env_group.add_argument('--lambda_var', type=float, default=1e-6,
                          help='Risk aversion')

    # Action space
    action_group = parser.add_argument_group('Action Space')
    action_group.add_argument('--n_action_bins', type=int, default=21,
                             help='Number of discrete action bins (0%%, 5%%, ..., 100%%)')

    # Info state
    info_group = parser.add_argument_group('Info State')
    info_group.add_argument('--info_state', type=str, default='lstm',
                           choices=['lstm', 'mlp'],
                           help='Info state encoder type')
    info_group.add_argument('--info_state_dim', type=int, default=64,
                           help='Info state dimension')
    info_group.add_argument('--lstm_embed_dim', type=int, default=128,
                           help='LSTM embedding dimension')
    info_group.add_argument('--lstm_num_layers', type=int, default=1,
                           help='Number of LSTM layers')

    # MAPPO hyperparameters
    mappo_group = parser.add_argument_group('MAPPO')
    mappo_group.add_argument('--lr', type=float, default=3e-4,
                            help='Learning rate')
    mappo_group.add_argument('--gamma', type=float, default=0.99,
                            help='Discount factor')
    mappo_group.add_argument('--gae_lambda', type=float, default=0.95,
                            help='GAE lambda')
    mappo_group.add_argument('--clip_epsilon', type=float, default=0.2,
                            help='PPO clip epsilon')
    mappo_group.add_argument('--value_coef', type=float, default=0.5,
                            help='Value loss coefficient')
    mappo_group.add_argument('--entropy_coef', type=float, default=0.01,
                            help='Entropy bonus coefficient')
    mappo_group.add_argument('--max_grad_norm', type=float, default=0.5,
                            help='Maximum gradient norm')
    mappo_group.add_argument('--n_ppo_epochs', type=int, default=4,
                            help='Number of PPO epochs per update')
    mappo_group.add_argument('--minibatch_size', type=int, default=64,
                            help='Minibatch size for updates')

    # Network
    net_group = parser.add_argument_group('Network')
    net_group.add_argument('--hidden_dim', type=int, default=128,
                          help='Hidden layer dimension')
    net_group.add_argument('--share_policy', action='store_true', default=True,
                          help='Share policy across agents')
    net_group.add_argument('--no_share_policy', action='store_false', dest='share_policy',
                          help='Independent policy per agent')
    net_group.add_argument('--share_value', action='store_true', default=True,
                          help='Share value function')
    net_group.add_argument('--no_share_value', action='store_false', dest='share_value',
                          help='Independent value per agent')
    net_group.add_argument('--centralized_critic', type=str, default='concat',
                          choices=['concat', 'global_state'],
                          help='Centralized critic input type')

    # Training
    train_group = parser.add_argument_group('Training')
    train_group.add_argument('--n_episodes', type=int, default=5000,
                            help='Number of training episodes')
    train_group.add_argument('--update_freq', type=int, default=10,
                            help='Update every N episodes')
    train_group.add_argument('--eval_freq', type=int, default=100,
                            help='Evaluate every N episodes')
    train_group.add_argument('--save_freq', type=int, default=500,
                            help='Save checkpoint every N episodes')
    train_group.add_argument('--save_path', type=str, default='./checkpoints/ac_mappo',
                            help='Checkpoint save path')

    # System
    sys_group = parser.add_argument_group('System')
    sys_group.add_argument('--device', type=str, default='cpu',
                          choices=['cpu', 'cuda', 'mps'],
                          help='Device to use')
    sys_group.add_argument('--seed', type=int, default=42,
                          help='Random seed')
    sys_group.add_argument('--no_progress', action='store_true',
                          help='Disable progress bar')
    sys_group.add_argument('--load_checkpoint', type=str, default=None,
                          help='Path to checkpoint to load')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
