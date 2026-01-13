"""A2C (Advantage Actor-Critic) agent."""

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim

from .base import Agent
from .networks import ActorCriticNetwork


class A2CAgent(Agent):
    """A2C agent with information state encoder."""

    def __init__(
        self,
        info_state_encoder,
        action_dim: int,
        config: Dict[str, Any],
    ):
        super().__init__(info_state_encoder, action_dim, config)

        state_dim = info_state_encoder.state_dim
        self.actor_critic = ActorCriticNetwork(
            state_dim,
            action_dim,
            hidden_dims=config.get('ac_hidden_dims', [128, 128]),
            shared_layers=config.get('shared_layers', 1),
        )

        self.gamma = config.get('gamma', 0.99)
        self.value_coef = config.get('value_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.lr = config.get('lr', 3e-4)

        # Single optimizer for all components
        self.optimizer = optim.Adam(
            list(self.info_state_encoder.parameters()) +
            list(self.actor_critic.parameters()),
            lr=self.lr
        )

    def get_action(
        self,
        obs: torch.Tensor,
        prev_action: Any = None,
        prev_reward: Any = None,
        hidden: Any = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Select action given observation.

        Args:
            obs: (batch, obs_dim)
            hidden: Hidden state
            deterministic: If True, return argmax

        Returns:
            action: (batch,)
            log_prob: (batch,)
            value: (batch, 1)
            hidden: Updated hidden state
        """
        # Get information state
        info_state, hidden = self.info_state_encoder(
            obs,
            action=prev_action,
            reward=prev_reward,
            hidden=hidden,
        )

        # Get action and value
        action, log_prob, value = self.actor_critic.get_action(info_state, deterministic)

        return action, log_prob, value, hidden

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        prev_actions: Optional[torch.Tensor] = None,
        prev_rewards: Optional[torch.Tensor] = None,
        hidden: Any = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions.

        Args:
            obs: (batch, obs_dim)
            actions: (batch,)
            prev_actions: For info state
            prev_rewards: For info state
            hidden: Hidden states

        Returns:
            log_probs: (batch,)
            values: (batch, 1)
            entropy: (batch,)
        """
        # Get information state
        info_state, _ = self.info_state_encoder(
            obs,
            action=prev_actions,
            reward=prev_rewards,
            hidden=hidden
        )

        # Evaluate
        log_probs, values, entropy = self.actor_critic.evaluate_actions(info_state, actions)

        return log_probs, values, entropy

    def update(self, rollout_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update using A2C.

        Args:
            rollout_data: Dictionary containing:
                - obs: (batch, seq_len, obs_dim)
                - actions: (batch, seq_len)
                - rewards: (batch, seq_len)
                - dones: (batch, seq_len)
                - values: (batch, seq_len) - old value estimates

        Returns:
            Dictionary of training metrics
        """
        obs = rollout_data['obs'].to(self.device)
        actions = rollout_data['actions'].to(self.device)
        rewards = rollout_data['rewards'].to(self.device)
        dones = rollout_data['dones'].to(self.device)
        old_values = rollout_data['values'].to(self.device)
        prev_actions = rollout_data.get('prev_actions')
        prev_rewards = rollout_data.get('prev_rewards')
        action_masks = rollout_data.get('action_masks')

        if prev_actions is not None:
            prev_actions = prev_actions.to(self.device)
        if prev_rewards is not None:
            prev_rewards = prev_rewards.to(self.device)
        if action_masks is None:
            action_masks = torch.ones_like(rewards, device=self.device)
        else:
            action_masks = action_masks.to(self.device)

        batch_size, seq_len = actions.shape

        # Compute returns and advantages
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)

        running_return = torch.zeros(batch_size, device=self.device)
        running_advantage = torch.zeros(batch_size, device=self.device)

        for t in reversed(range(seq_len)):
            running_return = rewards[:, t] + self.gamma * running_return * (1 - dones[:, t])
            returns[:, t] = running_return

            # TD error as advantage
            next_value = old_values[:, t + 1] if t + 1 < seq_len else 0
            td_error = rewards[:, t] + self.gamma * next_value * (1 - dones[:, t]) - old_values[:, t]
            advantages[:, t] = td_error

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Flatten
        if obs.dim() == 5:
            obs_flat = obs.reshape(batch_size * seq_len, *obs.shape[2:])
        else:
            obs_flat = obs.reshape(batch_size * seq_len, -1)
        actions_flat = actions.reshape(batch_size * seq_len)
        returns_flat = returns.reshape(batch_size * seq_len)
        advantages_flat = advantages.reshape(batch_size * seq_len)
        mask_flat = action_masks.reshape(batch_size * seq_len)

        prev_actions_flat = None
        prev_rewards_flat = None
        if prev_actions is not None:
            prev_actions_flat = prev_actions.reshape(batch_size * seq_len, -1)
        if prev_rewards is not None:
            prev_rewards_flat = prev_rewards.reshape(batch_size * seq_len, -1)

        # Evaluate current policy
        log_probs, values, entropy = self.evaluate_actions(
            obs_flat,
            actions_flat,
            prev_actions=prev_actions_flat,
            prev_rewards=prev_rewards_flat,
        )
        values = values.squeeze(-1)

        # Actor loss (policy gradient with advantage)
        mask_sum = mask_flat.sum().clamp_min(1.0)
        policy_loss = -(log_probs * advantages_flat * mask_flat).sum() / mask_sum

        # Critic loss (value function error)
        value_loss = ((values - returns_flat) ** 2 * mask_flat).sum() / mask_sum

        # Entropy bonus
        entropy_loss = -(entropy * mask_flat).sum() / mask_sum

        # Total loss
        total_loss = (
            policy_loss +
            self.value_coef * value_loss +
            self.entropy_coef * entropy_loss
        )

        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.config.get('max_grad_norm', 0.5))
        self.optimizer.step()

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.mean().item(),
            'total_loss': total_loss.item(),
            'mean_return': returns.mean().item(),
            'mean_advantage': advantages.mean().item(),
        }
