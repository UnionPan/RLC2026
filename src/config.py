"""Configuration for multi-agent training."""

from dataclasses import dataclass, field
from typing import Optional, List
import torch


def get_default_device() -> str:
    """Get default device (cuda if available, otherwise cpu)."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass
class InfoStateConfig:
    """Configuration for information state encoder."""
    type: str = 'lstm'  # 'lstm', 'cnn_lstm', 'mlp'
    state_dim: int = 64
    embed_dim: int = 128
    num_layers: int = 1
    cnn_channels: List[int] = field(default_factory=lambda: [16, 32])
    pretrained_path: Optional[str] = None
    freeze_pretrained: bool = False
    ae_latent_dim: int = 64
    ae_pretrained_path: Optional[str] = None
    ae_freeze: bool = False
    ae_transformer_dim: int = 128
    ae_transformer_heads: int = 4
    ae_transformer_layers: int = 2
    ae_transformer_ff_dim: int = 256
    ae_transformer_dropout: float = 0.1
    ae_max_seq_len: int = 64


@dataclass
class AgentConfig:
    """Configuration for agent."""
    type: str = 'a2c'  # 'reinforce', 'a2c', 'ppo'
    lr: float = 3e-4
    gamma: float = 0.99
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    policy_hidden_dims: List[int] = field(default_factory=lambda: [128, 128])
    ac_hidden_dims: List[int] = field(default_factory=lambda: [128, 128])
    shared_layers: int = 1


@dataclass
class EnvConfig:
    """Configuration for environment."""
    name: str = 'competitive_fourrooms'  # Environment name
    width: int = 11
    height: int = 11
    n_agents: int = 2
    view_radius: int = 2
    max_steps: int = 200
    wall_density: float = 0.1
    render_mode: Optional[str] = None


@dataclass
class TrainingConfig:
    """Configuration for training.

    Note: device defaults to 'cuda' but will automatically fall back to 'cpu'
    if CUDA is not available (handled in main.py).
    """
    n_episodes: int = 1000
    max_steps: int = 200
    update_freq: int = 10  # Update every N episodes
    eval_freq: int = 50
    save_freq: int = 100
    save_path: str = './checkpoints'
    device: str = 'cuda'  # Auto-fallback to CPU if CUDA unavailable
    seed: int = 42


@dataclass
class Config:
    """Main configuration."""
    info_state: InfoStateConfig = field(default_factory=InfoStateConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self):
        """Convert to dictionary."""
        return {
            'info_state': self.info_state.__dict__,
            'agent': self.agent.__dict__,
            'env': self.env.__dict__,
            'training': self.training.__dict__,
        }


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


def get_competitive_fourrooms_config() -> Config:
    """Configuration for competitive four rooms environment."""
    config = Config()
    config.env.name = 'competitive_fourrooms'
    config.env.width = 11
    config.env.height = 11
    config.env.n_agents = 2
    config.env.view_radius = 2
    config.env.max_steps = 200

    config.info_state.type = 'cnn_lstm'
    config.info_state.state_dim = 64
    config.info_state.cnn_channels = [16, 32]

    config.agent.type = 'a2c'
    config.agent.lr = 3e-4

    config.training.n_episodes = 2000
    config.training.update_freq = 10

    return config


def get_pursuit_evasion_config() -> Config:
    """Configuration for pursuit-evasion environment."""
    config = Config()
    config.env.name = 'pursuit_evasion'
    config.env.width = 15
    config.env.height = 15
    config.env.n_agents = 3  # 2 pursuers + 1 evader
    config.env.view_radius = 3
    config.env.max_steps = 100

    config.info_state.type = 'cnn_lstm'
    config.info_state.state_dim = 128
    config.info_state.cnn_channels = [32, 64]

    config.agent.type = 'a2c'
    config.agent.lr = 1e-4

    config.training.n_episodes = 5000
    config.training.update_freq = 20

    return config
