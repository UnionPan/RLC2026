"""Main entry point for multi-agent training."""

import argparse
import sys
import torch
import numpy as np

# Add lib to path
sys.path.insert(0, '/home/union/quant/RLC2026')

from lib.grid import competitive_envs, cooperatifve_envs, pursuit_evasion

from info_states import (
    LSTMInfoState,
    CNNLSTMInfoState,
    MLPInfoState,
    AutoencoderLSTMInfoState,
    AutoencoderTransformerInfoState,
)
from pg_agents import REINFORCEAgent, A2CAgent, PPOAgent
from pg_agents.mappo import MAPPOAgent
from pg_agents.qmix import QMIXAgent
from training import MultiAgentTrainer, ReplayBuffer
from config import (
    Config,
    get_default_config,
    get_competitive_fourrooms_config,
    get_pursuit_evasion_config,
)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_env(config: Config):
    """Create environment based on config."""
    env_name = config.env.name

    if env_name == 'competitive_fourrooms':
        env = competitive_envs.competitive_fourrooms_env(
            width=config.env.width,
            height=config.env.height,
            n_agents=config.env.n_agents,
            view_radius=config.env.view_radius,
            max_steps=config.env.max_steps,
            render_mode=config.env.render_mode,
        )
    elif env_name == 'competitive_obstructedmaze':
        env = competitive_envs.competitive_obstructedmaze_env(
            width=config.env.width,
            height=config.env.height,
            n_agents=config.env.n_agents,
            view_radius=config.env.view_radius,
            max_steps=config.env.max_steps,
            wall_density=config.env.wall_density,
            render_mode=config.env.render_mode,
        )

    # Cooperative environments
    elif env_name == 'cooperative_keycorridor':
        env = cooperatifve_envs.coop_keycorridor_env(
            width=config.env.width,
            height=config.env.height,
            n_agents=config.env.n_agents,
            view_radius=config.env.view_radius,
            max_steps=config.env.max_steps,
            render_mode=config.env.render_mode,
        )
    elif env_name == 'cooperative_lavacrossing':
        env = cooperatifve_envs.coop_lavacrossing_env(
            width=config.env.width,
            height=config.env.height,
            n_agents=config.env.n_agents,
            view_radius=config.env.view_radius,
            max_steps=config.env.max_steps,
            render_mode=config.env.render_mode,
        )

    # Pursuit-Evasion
    elif env_name == 'pursuit_evasion':
        env = pursuit_evasion.pursuit_evasion_env(
            width=config.env.width,
            height=config.env.height,
            n_pursuers=config.env.n_agents - 1,
            view_radius=config.env.view_radius,
            max_steps=config.env.max_steps,
        )
    else:
        raise ValueError(f"Unknown environment: {env_name}")

    return env


def create_info_state_encoder(config: Config, obs_shape: tuple, action_dim: int, device: str):
    """Create information state encoder based on config."""
    info_state_config = config.info_state

    if info_state_config.type == 'lstm':
        obs_dim = int(np.prod(obs_shape))
        encoder = LSTMInfoState(
            obs_dim=obs_dim,
            action_dim=action_dim,
            state_dim=info_state_config.state_dim,
            embed_dim=info_state_config.embed_dim,
            num_layers=info_state_config.num_layers,
        )
    elif info_state_config.type == 'cnn_lstm':
        encoder = CNNLSTMInfoState(
            obs_shape=obs_shape,
            action_dim=action_dim,
            state_dim=info_state_config.state_dim,
            cnn_channels=info_state_config.cnn_channels,
        )
    elif info_state_config.type == 'ae_lstm':
        encoder = AutoencoderLSTMInfoState(
            obs_shape=obs_shape,
            action_dim=action_dim,
            latent_dim=info_state_config.ae_latent_dim,
            state_dim=info_state_config.state_dim,
            num_layers=info_state_config.num_layers,
            pretrained_path=info_state_config.ae_pretrained_path,
            freeze_autoencoder=info_state_config.ae_freeze,
        )
    elif info_state_config.type == 'ae_transformer':
        encoder = AutoencoderTransformerInfoState(
            obs_shape=obs_shape,
            action_dim=action_dim,
            latent_dim=info_state_config.ae_latent_dim,
            model_dim=info_state_config.ae_transformer_dim,
            num_layers=info_state_config.ae_transformer_layers,
            num_heads=info_state_config.ae_transformer_heads,
            ff_dim=info_state_config.ae_transformer_ff_dim,
            dropout=info_state_config.ae_transformer_dropout,
            max_seq_len=info_state_config.ae_max_seq_len,
            pretrained_path=info_state_config.ae_pretrained_path,
            freeze_autoencoder=info_state_config.ae_freeze,
        )
    elif info_state_config.type == 'mlp':
        obs_dim = int(np.prod(obs_shape))
        encoder = MLPInfoState(
            obs_dim=obs_dim,
            action_dim=action_dim,
            state_dim=info_state_config.state_dim,
        )
    else:
        raise ValueError(f"Unknown info state type: {info_state_config.type}")

    encoder = encoder.to(device)
    if info_state_config.pretrained_path:
        state = torch.load(info_state_config.pretrained_path, map_location=device)
        encoder.load_state_dict(state, strict=False)
        if info_state_config.freeze_pretrained:
            for param in encoder.parameters():
                param.requires_grad = False

    return encoder


def create_agent(config: Config, obs_space, action_space, device: str):
    """Create agent based on config."""
    agent_config = config.agent
    obs_shape = obs_space.shape
    action_dim = action_space.n

    # Create info state encoder
    info_state_encoder = create_info_state_encoder(config, obs_shape, action_dim, device)

    # Prepare agent config dict
    agent_cfg = {
        'device': device,
        'lr': agent_config.lr,
        'gamma': agent_config.gamma,
        'entropy_coef': agent_config.entropy_coef,
        'max_grad_norm': agent_config.max_grad_norm,
    }

    # Create agent based on type
    if agent_config.type == 'reinforce':
        agent_cfg['policy_hidden_dims'] = agent_config.policy_hidden_dims
        agent = REINFORCEAgent(info_state_encoder, action_dim, agent_cfg)

    elif agent_config.type == 'a2c':
        agent_cfg['value_coef'] = agent_config.value_coef
        agent_cfg['ac_hidden_dims'] = agent_config.ac_hidden_dims
        agent_cfg['shared_layers'] = agent_config.shared_layers
        agent = A2CAgent(info_state_encoder, action_dim, agent_cfg)

    elif agent_config.type == 'ppo':
        agent_cfg['value_coef'] = agent_config.value_coef
        agent_cfg['clip_epsilon'] = getattr(agent_config, 'clip_epsilon', 0.2)
        agent_cfg['gae_lambda'] = getattr(agent_config, 'gae_lambda', 0.95)
        agent_cfg['normalize_advantages'] = getattr(agent_config, 'normalize_advantages', True)
        agent_cfg['use_clipped_value_loss'] = getattr(agent_config, 'use_clipped_value_loss', True)
        agent_cfg['hidden_dim'] = getattr(agent_config, 'hidden_dim', 128)
        agent = PPOAgent(obs_space, action_space, info_state_encoder, agent_cfg)

    else:
        raise ValueError(f"Unknown agent type: {agent_config.type}. "
                        f"Available: reinforce, a2c, ppo, mappo, qmix")

    return agent.to(device)


def main(args):
    """Main training function."""
    # Handle backward compatibility with --config
    if args.config is not None:
        print(f"⚠ Warning: --config is deprecated. Use --env and --algo instead.")
        if args.config == 'competitive_fourrooms':
            args.env = 'competitive_fourrooms'
            args.algo = 'a2c'
        elif args.config == 'pursuit_evasion':
            args.env = 'pursuit_evasion'
            args.algo = 'a2c'

    # Get default config
    config = get_default_config()

    # Set environment
    config.env.name = args.env

    # Set algorithm
    config.agent.type = args.algo

    # Environment-specific defaults
    if 'competitive' in args.env or 'cooperative' in args.env:
        config.env.n_agents = 2
        config.env.view_radius = 2
        config.env.max_steps = 200
        config.info_state.type = 'cnn_lstm'
        config.info_state.state_dim = 64
        config.info_state.cnn_channels = [16, 32, 32]
    elif args.env == 'pursuit_evasion':
        config.env.n_agents = 3  # 2 pursuers + 1 evader
        config.env.view_radius = 3
        config.env.max_steps = 100
        config.info_state.type = 'cnn_lstm'
        config.info_state.state_dim = 128
        config.info_state.cnn_channels = [32, 64]

    # Algorithm-specific defaults
    if args.algo == 'ppo':
        config.agent.lr = 3e-4
        config.training.update_freq = 10
        config.training.n_episodes = 2000
    elif args.algo == 'a2c':
        config.agent.lr = 3e-4
        config.training.update_freq = 10
        config.training.n_episodes = 2000
    elif args.algo == 'reinforce':
        config.agent.lr = 1e-3
        config.training.update_freq = 1
        config.training.n_episodes = 3000
    elif args.algo in ['mappo', 'qmix']:
        config.agent.lr = 5e-4
        config.training.update_freq = 20
        config.training.n_episodes = 5000

    # Override with command line args
    if args.episodes is not None:
        config.training.n_episodes = args.episodes
    if args.lr is not None:
        config.agent.lr = args.lr
    if args.gamma is not None:
        config.agent.gamma = args.gamma
    if args.device is not None:
        config.training.device = args.device
    if args.seed is not None:
        config.training.seed = args.seed

    # Set save path based on environment and algorithm
    config.training.save_path = f'./checkpoints/{args.env}_{args.algo}'

    # Set device with automatic fallback
    device = config.training.device
    if device == 'cuda':
        if torch.cuda.is_available():
            # GPU available - show info
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"\n✓ GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            # CUDA requested but not available - fallback to CPU
            print("\n⚠ CUDA not available, falling back to CPU")
            device = 'cpu'
            config.training.device = 'cpu'
    else:
        print(f"\n• Using device: {device}")

    # Set seed
    set_seed(config.training.seed)

    print("="*60)
    print("Multi-Agent RL Training")
    print("="*60)
    print(f"Environment: {config.env.name}")
    print(f"Info State: {config.info_state.type}")
    print(f"Agent: {config.agent.type}")
    print(f"Device: {device}")
    print(f"Episodes: {config.training.n_episodes}")
    print("="*60)

    # Create environment
    env = create_env(config)
    env.reset()

    # Get environment info
    sample_agent = env.possible_agents[0]
    obs_space = env.observation_space(sample_agent)
    act_space = env.action_space(sample_agent)
    obs_shape = obs_space.shape
    action_dim = act_space.n

    print(f"Observation shape: {obs_shape}")
    print(f"Action space: {action_dim}")
    print(f"Agents: {env.possible_agents}")
    print("="*60)

    # Create agents (one per agent in env)
    agents = {}
    for agent_id in env.possible_agents:
        agent = create_agent(config, obs_space, act_space, device)
        agents[agent_id] = agent

    print(f"Created {len(agents)} agents ({config.agent.type.upper()})")

    # Create trainer
    trainer_config = {
        'device': device,
        'n_episodes': config.training.n_episodes,
        'max_steps': config.env.max_steps,
        'update_freq': config.training.update_freq,
        'eval_freq': config.training.eval_freq,
        'save_freq': config.training.save_freq,
        'save_path': config.training.save_path,
    }

    trainer = MultiAgentTrainer(env, agents, trainer_config)

    # Train
    episode_rewards, training_metrics = trainer.train()

    # Final evaluation
    print("\nFinal Evaluation (100 episodes)...")
    final_stats = trainer.evaluate(n_episodes=100)
    print(final_stats)

    print("\nTraining completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-Agent RL Training')

    # Environment selection
    parser.add_argument('--env', '--environment', type=str, default='competitive_fourrooms',
                        choices=[
                            'competitive_fourrooms',
                            'competitive_obstructedmaze',
                            'cooperative_keycorridor',
                            'cooperative_lavacrossing',
                            'pursuit_evasion'
                        ],
                        help='Environment to train on')

    # Algorithm selection
    parser.add_argument('--algo', '--algorithm', type=str, default='a2c',
                        choices=['reinforce', 'a2c', 'ppo', 'mappo', 'qmix'],
                        help='RL algorithm to use')

    # Deprecated config argument (for backward compatibility)
    parser.add_argument('--config', type=str, default=None,
                        help='DEPRECATED: Use --env and --algo instead')

    # Training parameters
    parser.add_argument('--episodes', type=int, default=None,
                        help='Number of training episodes')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=None,
                        help='Discount factor')

    # System parameters
    parser.add_argument('--device', type=str, default=None,
                        choices=['cpu', 'cuda'],
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')

    args = parser.parse_args()
    main(args)
