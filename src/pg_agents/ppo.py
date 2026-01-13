"""Proximal Policy Optimization (PPO) agent."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Dict, Optional, Tuple, Any
import numpy as np

from .base import Agent
from .networks import ActorCriticNetwork


class PPOAgent(Agent):
    """PPO agent with clipped objective and value function learning."""

    def __init__(
        self,
        obs_space,
        action_space,
        info_state_encoder,
        config: Dict[str, Any],
    ):
        """
        Args:
            obs_space: Observation space
            action_space: Action space
            info_state_encoder: InfoState encoder module
            config: Configuration dict with:
                - lr: Learning rate (default: 3e-4)
                - gamma: Discount factor (default: 0.99)
                - gae_lambda: GAE lambda (default: 0.95)
                - clip_epsilon: PPO clip parameter (default: 0.2)
                - value_coef: Value loss coefficient (default: 0.5)
                - entropy_coef: Entropy bonus coefficient (default: 0.01)
                - max_grad_norm: Gradient clipping (default: 0.5)
                - normalize_advantages: Whether to normalize advantages (default: True)
                - use_clipped_value_loss: Use clipped value loss (default: True)
                - device: torch device
        """
        super().__init__(obs_space, action_space, info_state_encoder, config)

        self.lr = config.get('lr', 3e-4)
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        self.value_coef = config.get('value_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.normalize_advantages = config.get('normalize_advantages', True)
        self.use_clipped_value_loss = config.get('use_clipped_value_loss', True)
        self.device = config.get('device', 'cpu')

        # Get state dimension from info state encoder
        state_dim = self.info_state_encoder.state_dim
        n_actions = action_space.n

        # Actor-Critic network
        self.ac_network = ActorCriticNetwork(
            state_dim=state_dim,
            n_actions=n_actions,
            hidden_dim=config.get('hidden_dim', 128),
        ).to(self.device)

        # Optimizer for both policy and value function
        self.optimizer = torch.optim.Adam(
            list(self.info_state_encoder.parameters()) + list(self.ac_network.parameters()),
            lr=self.lr,
        )

        # Hidden state for recurrent info state encoder
        self.hidden = None

    def select_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        action_prev: Optional[torch.Tensor] = None,
        reward_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """Select action using current policy.

        Args:
            obs: Observation tensor
            deterministic: If True, select argmax action
            action_prev: Previous action (for info state)
            reward_prev: Previous reward (for info state)

        Returns:
            action: Selected action (int)
            info: Dict with 'log_prob', 'value', 'entropy', 'hidden'
        """
        with torch.no_grad():
            # Encode observation to info state
            info_state, self.hidden = self.info_state_encoder(
                obs, action_prev, reward_prev, self.hidden
            )

            # Get policy and value
            action_logits, value = self.ac_network(info_state)
            probs = F.softmax(action_logits, dim=-1)
            dist = Categorical(probs)

            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                action = dist.sample()

            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

        return action.item(), {
            'log_prob': log_prob.item(),
            'value': value.item(),
            'entropy': entropy.item(),
            'hidden': self.hidden,
        }

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        actions_prev: Optional[torch.Tensor] = None,
        rewards_prev: Optional[torch.Tensor] = None,
        hidden_states: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update.

        Args:
            obs: Observation batch (batch, obs_dim)
            actions: Action batch (batch,)
            actions_prev: Previous actions (batch, action_dim)
            rewards_prev: Previous rewards (batch, 1)
            hidden_states: Hidden states from rollout

        Returns:
            log_probs: Log probabilities (batch,)
            values: State values (batch,)
            entropy: Entropy (batch,)
        """
        # Encode observations to info states
        info_states, _ = self.info_state_encoder(
            obs, actions_prev, rewards_prev, hidden_states
        )

        # Get policy and value
        action_logits, values = self.ac_network(info_states)
        probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(probs)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, values.squeeze(-1), entropy

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: Reward sequence (T,)
            values: Value predictions (T,)
            dones: Done flags (T,)
            next_value: Bootstrap value (scalar)

        Returns:
            advantages: GAE advantages (T,)
            returns: Discounted returns (T,)
        """
        T = len(rewards)
        advantages = torch.zeros(T, device=self.device)
        gae = 0

        # Backward pass for GAE
        for t in reversed(range(T)):
            if t == T - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def update(
        self,
        rollout_buffer,
        n_epochs: int = 10,
        batch_size: int = 64,
    ) -> Dict[str, float]:
        """Update policy using PPO objective.

        Args:
            rollout_buffer: RolloutBuffer with trajectory data
            n_epochs: Number of update epochs
            batch_size: Minibatch size

        Returns:
            Dict with loss statistics
        """
        # Compute advantages using GAE
        advantages, returns = self.compute_gae(
            rollout_buffer.rewards,
            rollout_buffer.values,
            rollout_buffer.dones,
            rollout_buffer.next_value,
        )

        # Normalize advantages
        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Store for later use
        rollout_buffer.advantages = advantages
        rollout_buffer.returns = returns

        # Training statistics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_clipfrac = 0
        n_updates = 0

        # Multiple epochs over the same data
        for epoch in range(n_epochs):
            # Generate random minibatches
            indices = np.arange(len(rollout_buffer))
            np.random.shuffle(indices)

            for start in range(0, len(rollout_buffer), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]

                # Get batch data
                obs_batch = rollout_buffer.observations[batch_indices]
                actions_batch = rollout_buffer.actions[batch_indices]
                old_log_probs_batch = rollout_buffer.log_probs[batch_indices]
                advantages_batch = advantages[batch_indices]
                returns_batch = returns[batch_indices]
                old_values_batch = rollout_buffer.values[batch_indices]

                # Get previous actions/rewards for info state
                actions_prev_batch = None
                rewards_prev_batch = None
                if hasattr(rollout_buffer, 'actions_prev'):
                    actions_prev_batch = rollout_buffer.actions_prev[batch_indices]
                    rewards_prev_batch = rollout_buffer.rewards_prev[batch_indices]

                # Evaluate current policy on old states
                log_probs, values, entropy = self.evaluate_actions(
                    obs_batch,
                    actions_batch,
                    actions_prev_batch,
                    rewards_prev_batch,
                    None,  # Don't use stored hidden states for batch training
                )

                # PPO policy loss
                ratio = torch.exp(log_probs - old_log_probs_batch)
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_batch
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                if self.use_clipped_value_loss:
                    value_pred_clipped = old_values_batch + torch.clamp(
                        values - old_values_batch,
                        -self.clip_epsilon,
                        self.clip_epsilon,
                    )
                    value_loss_unclipped = F.mse_loss(values, returns_batch)
                    value_loss_clipped = F.mse_loss(value_pred_clipped, returns_batch)
                    value_loss = torch.max(value_loss_unclipped, value_loss_clipped)
                else:
                    value_loss = F.mse_loss(values, returns_batch)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.info_state_encoder.parameters()) + list(self.ac_network.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                # Track statistics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                with torch.no_grad():
                    clipfrac = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean()
                    total_clipfrac += clipfrac.item()
                n_updates += 1

        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates,
            'clipfrac': total_clipfrac / n_updates,
        }

    def reset_hidden(self):
        """Reset hidden state for new episode."""
        self.hidden = None

    def save(self, path: str):
        """Save agent checkpoint."""
        torch.save({
            'info_state_encoder': self.info_state_encoder.state_dict(),
            'ac_network': self.ac_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        """Load agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.info_state_encoder.load_state_dict(checkpoint['info_state_encoder'])
        self.ac_network.load_state_dict(checkpoint['ac_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])