"""QMIX: Monotonic Value Function Factorisation for Decentralised Multi-Agent RL."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .base import Agent
from .networks import QNetwork


class QMixingNetwork(nn.Module):
    """Mixing network that combines individual Q-values into global Q.

    Uses absolute weights to ensure monotonicity: dQ_tot/dQ_i >= 0
    """

    def __init__(
        self,
        num_agents: int,
        state_dim: int,
        hidden_dim: int = 32,
        hypernet_hidden_dim: int = 64,
    ):
        """
        Args:
            num_agents: Number of agents
            state_dim: Global state dimension
            hidden_dim: Hidden layer size for mixing network
            hypernet_hidden_dim: Hidden layer size for hypernetworks
        """
        super().__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # Hypernetwork for weights of first layer
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_hidden_dim),
            nn.ReLU(),
            nn.Linear(hypernet_hidden_dim, num_agents * hidden_dim),
        )

        # Hypernetwork for bias of first layer
        self.hyper_b1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
        )

        # Hypernetwork for weights of second layer
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_hidden_dim),
            nn.ReLU(),
            nn.Linear(hypernet_hidden_dim, hidden_dim),
        )

        # Hypernetwork for bias of second layer (scalar)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_hidden_dim),
            nn.ReLU(),
            nn.Linear(hypernet_hidden_dim, 1),
        )

    def forward(self, q_values: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Mix individual Q-values into global Q-value.

        Args:
            q_values: Individual Q-values (batch, num_agents)
            state: Global state (batch, state_dim)

        Returns:
            q_total: Mixed Q-value (batch, 1)
        """
        batch_size = q_values.shape[0]

        # Generate weights and biases from hypernetworks
        w1 = torch.abs(self.hyper_w1(state))  # Absolute for monotonicity
        b1 = self.hyper_b1(state)
        w2 = torch.abs(self.hyper_w2(state))  # Absolute for monotonicity
        b2 = self.hyper_b2(state)

        # Reshape weights for mixing
        w1 = w1.view(batch_size, self.num_agents, self.hidden_dim)
        b1 = b1.view(batch_size, 1, self.hidden_dim)
        w2 = w2.view(batch_size, self.hidden_dim, 1)
        b2 = b2.view(batch_size, 1, 1)

        # First layer: (batch, 1, num_agents) @ (batch, num_agents, hidden_dim)
        q_values = q_values.view(batch_size, 1, self.num_agents)
        hidden = F.elu(torch.bmm(q_values, w1) + b1)

        # Second layer: (batch, 1, hidden_dim) @ (batch, hidden_dim, 1)
        q_total = torch.bmm(hidden, w2) + b2

        return q_total.view(batch_size, 1)


class QMIXAgent:
    """QMIX agent with value function factorization."""

    def __init__(
        self,
        obs_space,
        action_space,
        num_agents: int,
        info_state_encoder_fn,
        config: Dict[str, Any],
    ):
        """
        Args:
            obs_space: Observation space
            action_space: Action space
            num_agents: Number of agents
            info_state_encoder_fn: Function to create info state encoder
            config: Configuration dict with:
                - global_state_dim: Dimension of global state
                - share_q_network: Share Q-network across agents (default: False)
                - lr: Learning rate (default: 5e-4)
                - gamma: Discount factor (default: 0.99)
                - epsilon_start: Initial exploration (default: 1.0)
                - epsilon_end: Final exploration (default: 0.05)
                - epsilon_decay: Decay steps (default: 50000)
                - target_update_interval: Steps between target updates (default: 200)
                - mixing_hidden_dim: Hidden dim for mixing network (default: 32)
                - double_q: Use double Q-learning (default: True)
                - device: torch device
        """
        self.num_agents = num_agents
        self.obs_space = obs_space
        self.action_space = action_space
        self.config = config
        self.device = config.get('device', 'cpu')

        self.n_actions = action_space.n
        self.share_q_network = config.get('share_q_network', False)
        self.global_state_dim = config.get('global_state_dim', 64)

        # Q-learning parameters
        self.lr = config.get('lr', 5e-4)
        self.gamma = config.get('gamma', 0.99)
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_start = config.get('epsilon_start', 1.0)
        self.epsilon_end = config.get('epsilon_end', 0.05)
        self.epsilon_decay = config.get('epsilon_decay', 50000)
        self.target_update_interval = config.get('target_update_interval', 200)
        self.double_q = config.get('double_q', True)

        self.steps = 0

        # Create info state encoders
        if self.share_q_network:
            shared_encoder = info_state_encoder_fn()
            self.info_state_encoders = {i: shared_encoder for i in range(num_agents)}
        else:
            self.info_state_encoders = {
                i: info_state_encoder_fn() for i in range(num_agents)
            }

        state_dim = list(self.info_state_encoders.values())[0].state_dim

        # Create Q-networks (online)
        if self.share_q_network:
            shared_q = QNetwork(
                state_dim=state_dim,
                n_actions=self.n_actions,
                hidden_dim=config.get('hidden_dim', 128),
            ).to(self.device)
            self.q_networks = {i: shared_q for i in range(num_agents)}
        else:
            self.q_networks = {
                i: QNetwork(
                    state_dim=state_dim,
                    n_actions=self.n_actions,
                    hidden_dim=config.get('hidden_dim', 128),
                ).to(self.device)
                for i in range(num_agents)
            }

        # Create target Q-networks
        if self.share_q_network:
            shared_target_q = QNetwork(
                state_dim=state_dim,
                n_actions=self.n_actions,
                hidden_dim=config.get('hidden_dim', 128),
            ).to(self.device)
            shared_target_q.load_state_dict(shared_q.state_dict())
            self.target_q_networks = {i: shared_target_q for i in range(num_agents)}
        else:
            self.target_q_networks = {}
            for i in range(num_agents):
                target_q = QNetwork(
                    state_dim=state_dim,
                    n_actions=self.n_actions,
                    hidden_dim=config.get('hidden_dim', 128),
                ).to(self.device)
                target_q.load_state_dict(self.q_networks[i].state_dict())
                self.target_q_networks[i] = target_q

        # Create mixing networks
        self.mixer = QMixingNetwork(
            num_agents=num_agents,
            state_dim=self.global_state_dim,
            hidden_dim=config.get('mixing_hidden_dim', 32),
        ).to(self.device)

        self.target_mixer = QMixingNetwork(
            num_agents=num_agents,
            state_dim=self.global_state_dim,
            hidden_dim=config.get('mixing_hidden_dim', 32),
        ).to(self.device)
        self.target_mixer.load_state_dict(self.mixer.state_dict())

        # Optimizer
        unique_params = []
        seen_ids = set()

        for encoder in self.info_state_encoders.values():
            if id(encoder) not in seen_ids:
                unique_params.extend(list(encoder.parameters()))
                seen_ids.add(id(encoder))

        for q_net in self.q_networks.values():
            if id(q_net) not in seen_ids:
                unique_params.extend(list(q_net.parameters()))
                seen_ids.add(id(q_net))

        unique_params.extend(list(self.mixer.parameters()))

        self.optimizer = torch.optim.Adam(unique_params, lr=self.lr)

        # Hidden states
        self.hidden_states = {i: None for i in range(num_agents)}

    def select_actions(
        self,
        observations: Dict[int, torch.Tensor],
        epsilon: Optional[float] = None,
        actions_prev: Optional[Dict[int, torch.Tensor]] = None,
        rewards_prev: Optional[Dict[int, torch.Tensor]] = None,
    ) -> Tuple[Dict[int, int], Dict[int, Dict[str, Any]]]:
        """Select actions using epsilon-greedy policy.

        Args:
            observations: Dict {agent_id: observation}
            epsilon: Exploration rate (uses self.epsilon if None)
            actions_prev: Previous actions
            rewards_prev: Previous rewards

        Returns:
            actions: Dict {agent_id: action}
            infos: Dict {agent_id: info_dict with q_values}
        """
        if epsilon is None:
            epsilon = self.epsilon

        with torch.no_grad():
            actions = {}
            infos = {}

            for i, obs in observations.items():
                action_prev = actions_prev[i] if actions_prev else None
                reward_prev = rewards_prev[i] if rewards_prev else None

                # Get info state
                info_state, self.hidden_states[i] = self.info_state_encoders[i](
                    obs, action_prev, reward_prev, self.hidden_states[i]
                )

                # Get Q-values
                q_values = self.q_networks[i](info_state)

                # Epsilon-greedy action selection
                if np.random.random() < epsilon:
                    action = np.random.randint(0, self.n_actions)
                else:
                    action = torch.argmax(q_values, dim=-1).item()

                actions[i] = action
                infos[i] = {
                    'q_values': q_values.cpu().numpy(),
                    'info_state': info_state.detach(),
                    'hidden': self.hidden_states[i],
                }

        return actions, infos

    def update(
        self,
        batch: Dict[str, Any],
    ) -> Dict[str, float]:
        """Update Q-networks and mixer using QMIX loss.

        Args:
            batch: Dict with:
                - observations: Dict {agent_id: (batch, obs_dim)}
                - actions: Dict {agent_id: (batch,)}
                - rewards: (batch,) - shared team reward
                - next_observations: Dict {agent_id: (batch, obs_dim)}
                - dones: (batch,)
                - global_states: (batch, global_state_dim)
                - next_global_states: (batch, global_state_dim)

        Returns:
            Dict with loss statistics
        """
        observations = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']
        next_observations = batch['next_observations']
        dones = batch['dones']
        global_states = batch['global_states']
        next_global_states = batch['next_global_states']

        batch_size = rewards.shape[0]

        # Get current Q-values for chosen actions
        chosen_q_values = []
        for i in range(self.num_agents):
            # Encode observations
            info_state, _ = self.info_state_encoders[i](observations[i], None, None, None)

            # Get Q-values
            q_vals = self.q_networks[i](info_state)

            # Select Q-values for actions taken
            chosen_q = q_vals.gather(1, actions[i].unsqueeze(1))
            chosen_q_values.append(chosen_q)

        # Stack: (batch, num_agents)
        chosen_q_values = torch.cat(chosen_q_values, dim=1)

        # Mix chosen Q-values
        q_total = self.mixer(chosen_q_values, global_states)

        # Get target Q-values
        with torch.no_grad():
            if self.double_q:
                # Double Q-learning: use online network to select actions
                next_q_values = []
                for i in range(self.num_agents):
                    info_state, _ = self.info_state_encoders[i](next_observations[i], None, None, None)
                    q_vals = self.q_networks[i](info_state)
                    next_q_values.append(q_vals)

                # Select best actions
                next_actions = [torch.argmax(q, dim=1) for q in next_q_values]

                # Use target network to evaluate
                target_q_values = []
                for i in range(self.num_agents):
                    info_state, _ = self.info_state_encoders[i](next_observations[i], None, None, None)
                    target_q_vals = self.target_q_networks[i](info_state)
                    target_q = target_q_vals.gather(1, next_actions[i].unsqueeze(1))
                    target_q_values.append(target_q)
            else:
                # Standard Q-learning: use target network for both selection and evaluation
                target_q_values = []
                for i in range(self.num_agents):
                    info_state, _ = self.info_state_encoders[i](next_observations[i], None, None, None)
                    target_q_vals = self.target_q_networks[i](info_state)
                    target_q_max = target_q_vals.max(dim=1, keepdim=True)[0]
                    target_q_values.append(target_q_max)

            # Stack and mix
            target_q_values = torch.cat(target_q_values, dim=1)
            target_q_total = self.target_mixer(target_q_values, next_global_states)

            # Compute target
            targets = rewards.unsqueeze(1) + self.gamma * (1 - dones.unsqueeze(1)) * target_q_total

        # TD loss
        loss = F.mse_loss(q_total, targets)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], 10.0)
        self.optimizer.step()

        # Update target networks periodically
        self.steps += 1
        if self.steps % self.target_update_interval == 0:
            self.update_target_networks()

        # Decay epsilon
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start - (self.epsilon_start - self.epsilon_end) * self.steps / self.epsilon_decay
        )

        return {
            'loss': loss.item(),
            'q_mean': q_total.mean().item(),
            'target_mean': targets.mean().item(),
            'epsilon': self.epsilon,
        }

    def update_target_networks(self):
        """Update target networks with current network weights."""
        # Update target Q-networks
        unique_updated = set()
        for i in range(self.num_agents):
            if id(self.q_networks[i]) not in unique_updated:
                self.target_q_networks[i].load_state_dict(self.q_networks[i].state_dict())
                unique_updated.add(id(self.q_networks[i]))

        # Update target mixer
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def reset_hidden(self):
        """Reset hidden states for all agents."""
        for i in range(self.num_agents):
            self.hidden_states[i] = None

    def save(self, path: str):
        """Save checkpoint."""
        checkpoint = {
            'mixer': self.mixer.state_dict(),
            'target_mixer': self.target_mixer.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
        }

        # Save unique networks
        saved_ids = set()
        for i, encoder in enumerate(self.info_state_encoders.values()):
            if id(encoder) not in saved_ids:
                checkpoint[f'info_state_encoder_{i}'] = encoder.state_dict()
                saved_ids.add(id(encoder))

        saved_ids = set()
        for i, q_net in enumerate(self.q_networks.values()):
            if id(q_net) not in saved_ids:
                checkpoint[f'q_network_{i}'] = q_net.state_dict()
                checkpoint[f'target_q_network_{i}'] = self.target_q_networks[i].state_dict()
                saved_ids.add(id(q_net))

        torch.save(checkpoint, path)

    def load(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.mixer.load_state_dict(checkpoint['mixer'])
        self.target_mixer.load_state_dict(checkpoint['target_mixer'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']

        # Load unique networks
        loaded_ids = {}
        for key in checkpoint:
            if key.startswith('info_state_encoder_'):
                idx = int(key.split('_')[-1])
                encoder = list(self.info_state_encoders.values())[idx]
                if id(encoder) not in loaded_ids:
                    encoder.load_state_dict(checkpoint[key])
                    loaded_ids[id(encoder)] = True

            elif key.startswith('q_network_'):
                idx = int(key.split('_')[-1])
                q_net = list(self.q_networks.values())[idx]
                target_q_net = list(self.target_q_networks.values())[idx]
                if id(q_net) not in loaded_ids:
                    q_net.load_state_dict(checkpoint[key])
                    target_key = f'target_q_network_{idx}'
                    if target_key in checkpoint:
                        target_q_net.load_state_dict(checkpoint[target_key])
                    loaded_ids[id(q_net)] = True
