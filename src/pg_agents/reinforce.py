"""REINFORCE agent with information state."""

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim

from .base import Agent
from .networks import PolicyNetwork


class REINFORCEAgent(Agent):
    """REINFORCE policy gradient agent."""

    def __init__(
        self,
        info_state_encoder,
        action_dim: int,
        config: Dict[str, Any],
    ):
        super().__init__(info_state_encoder, action_dim, config)

        state_dim = info_state_encoder.state_dim
        self.policy = PolicyNetwork(
            state_dim,
            action_dim,
            hidden_dims=config.get('policy_hidden_dims', [128, 128])
        )

        self.gamma = config.get('gamma', 0.99)
        self.lr = config.get('lr', 3e-4)

        # Optimizer for both info state encoder and policy
        self.optimizer = optim.Adam(
            list(self.info_state_encoder.parameters()) + list(self.policy.parameters()),
            lr=self.lr
        )

    def get_action(
        self,
        obs: torch.Tensor,
        prev_action: Any = None,
        prev_reward: Any = None,
        hidden: Any = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """Select action given observation.

        Args:
            obs: (batch, obs_dim)
            hidden: Hidden state for info state encoder
            deterministic: If True, return argmax action

        Returns:
            action: (batch,)
            log_prob: (batch,)
            hidden: Updated hidden state
        """
        # Get information state
        info_state, hidden = self.info_state_encoder(
            obs,
            action=prev_action,
            reward=prev_reward,
            hidden=hidden,
        )

        # Get action from policy
        action, log_prob = self.policy.get_action(info_state, deterministic)

        return action, log_prob, hidden

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        prev_actions: Optional[torch.Tensor] = None,
        prev_rewards: Optional[torch.Tensor] = None,
        hidden: Any = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate actions for training.

        Args:
            obs: (batch, obs_dim) or (batch, seq_len, obs_dim)
            actions: (batch,) or (batch, seq_len)
            prev_actions: Previous actions for info state
            prev_rewards: Previous rewards for info state
            hidden: Hidden states

        Returns:
            log_probs: (batch,)
            entropy: (batch,)
        """
        # Get information state
        info_state, _ = self.info_state_encoder(
            obs,
            action=prev_actions,
            reward=prev_rewards,
            hidden=hidden
        )

        # Evaluate actions
        log_probs, entropy = self.policy.evaluate_actions(info_state, actions)

        return log_probs, entropy

    def update(self, rollout_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update policy using REINFORCE.

        Args:
            rollout_data: Dictionary containing:
                - obs: (batch, seq_len, obs_dim)
                - actions: (batch, seq_len)
                - rewards: (batch, seq_len)
                - dones: (batch, seq_len)
                - log_probs: (batch, seq_len) - old log probs

        Returns:
            Dictionary of training metrics
        """
        obs = rollout_data['obs'].to(self.device)
        actions = rollout_data['actions'].to(self.device)
        rewards = rollout_data['rewards'].to(self.device)
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

        # Compute returns (discounted cumulative rewards)
        returns = torch.zeros_like(rewards)
        running_return = torch.zeros(batch_size, device=self.device)

        for t in reversed(range(seq_len)):
            running_return = rewards[:, t] + self.gamma * running_return
            returns[:, t] = running_return

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Flatten for processing
        if obs.dim() == 5:
            obs_flat = obs.reshape(batch_size * seq_len, *obs.shape[2:])
        else:
            obs_flat = obs.reshape(batch_size * seq_len, -1)
        actions_flat = actions.reshape(batch_size * seq_len)
        returns_flat = returns.reshape(batch_size * seq_len)
        mask_flat = action_masks.reshape(batch_size * seq_len)

        prev_actions_flat = None
        prev_rewards_flat = None
        if prev_actions is not None:
            prev_actions_flat = prev_actions.reshape(batch_size * seq_len, -1)
        if prev_rewards is not None:
            prev_rewards_flat = prev_rewards.reshape(batch_size * seq_len, -1)

        # Get log probs and entropy
        log_probs, entropy = self.evaluate_actions(
            obs_flat,
            actions_flat,
            prev_actions=prev_actions_flat,
            prev_rewards=prev_rewards_flat,
        )

        # REINFORCE loss: -E[log π(a|s) * R]
        mask_sum = mask_flat.sum().clamp_min(1.0)
        policy_loss = -(log_probs * returns_flat * mask_flat).sum() / mask_sum

        # Optional entropy bonus
        entropy_coef = self.config.get('entropy_coef', 0.01)
        entropy_loss = -(entropy * mask_flat).sum() / mask_sum

        total_loss = policy_loss + entropy_coef * entropy_loss

        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.config.get('max_grad_norm', 0.5))
        self.optimizer.step()

        return {
            'policy_loss': policy_loss.item(),
            'entropy': entropy.mean().item(),
            'total_loss': total_loss.item(),
            'mean_return': returns.mean().item(),
        }
