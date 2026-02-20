"""Proximal Policy Optimization (PPO) agent."""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .base import Agent
from .networks import ActorCriticNetwork


class PPOAgent(Agent):
    """PPO agent integrated with the multi-agent rollout trainer."""

    def __init__(
        self,
        obs_space,
        action_space,
        info_state_encoder,
        config: Dict[str, Any],
    ):
        del obs_space  # Unused; retained for API compatibility.
        super().__init__(info_state_encoder, action_space.n, config)

        self.lr = config.get('lr', 3e-4)
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        self.value_coef = config.get('value_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.normalize_advantages = config.get('normalize_advantages', True)
        self.use_clipped_value_loss = config.get('use_clipped_value_loss', True)
        self.ppo_epochs = config.get('ppo_epochs', 4)
        self.batch_size = config.get('batch_size', 256)

        hidden_dims = config.get('ac_hidden_dims')
        if hidden_dims is None:
            hidden_dim = config.get('hidden_dim', 128)
            hidden_dims = [hidden_dim, hidden_dim]

        self.ac_network = ActorCriticNetwork(
            state_dim=self.info_state_encoder.state_dim,
            action_dim=self.action_dim,
            hidden_dims=hidden_dims,
            shared_layers=config.get('shared_layers', 1),
        )

        self.optimizer = torch.optim.Adam(
            list(self.info_state_encoder.parameters()) + list(self.ac_network.parameters()),
            lr=self.lr,
        )

    def get_action(
        self,
        obs: torch.Tensor,
        prev_action: Any = None,
        prev_reward: Any = None,
        hidden: Any = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Select action given observation."""
        info_state, hidden = self.info_state_encoder(
            obs,
            action=prev_action,
            reward=prev_reward,
            hidden=hidden,
        )
        action, log_prob, value = self.ac_network.get_action(info_state, deterministic)
        return action, log_prob, value, hidden

    def select_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        action_prev: Optional[torch.Tensor] = None,
        reward_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """Compatibility wrapper around `get_action`."""
        action, log_prob, value, hidden = self.get_action(
            obs,
            prev_action=action_prev,
            prev_reward=reward_prev,
            hidden=None,
            deterministic=deterministic,
        )
        return int(action.item()), {
            'log_prob': float(log_prob.item()),
            'value': float(value.item()),
            'hidden': hidden,
        }

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        prev_actions: Optional[torch.Tensor] = None,
        prev_rewards: Optional[torch.Tensor] = None,
        hidden: Any = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions under current policy."""
        info_states, _ = self.info_state_encoder(
            obs,
            action=prev_actions,
            reward=prev_rewards,
            hidden=hidden,
        )
        log_probs, values, entropy = self.ac_network.evaluate_actions(info_states, actions)
        return log_probs, values.squeeze(-1), entropy

    def _compute_returns_advantages(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute per-step returns and GAE advantages for padded rollouts."""
        batch_size, seq_len = rewards.shape
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)

        next_return = torch.zeros(batch_size, device=self.device)
        next_advantage = torch.zeros(batch_size, device=self.device)
        next_value = torch.zeros(batch_size, device=self.device)

        for t in reversed(range(seq_len)):
            not_done = 1.0 - dones[:, t]
            next_return = rewards[:, t] + self.gamma * next_return * not_done
            returns[:, t] = next_return

            delta = rewards[:, t] + self.gamma * next_value * not_done - values[:, t]
            next_advantage = delta + self.gamma * self.gae_lambda * not_done * next_advantage
            advantages[:, t] = next_advantage
            next_value = values[:, t]

        return returns, advantages

    def update(self, rollout_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update policy and value function with PPO clipped objective."""
        obs = rollout_data['obs'].to(self.device)
        actions = rollout_data['actions'].to(self.device)
        rewards = rollout_data['rewards'].to(self.device)
        dones = rollout_data['dones'].to(self.device)
        old_log_probs = rollout_data['log_probs'].to(self.device)
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

        returns, advantages = self._compute_returns_advantages(rewards, dones, old_values)

        valid_advantages = advantages[action_masks > 0.0]
        if self.normalize_advantages and valid_advantages.numel() > 1:
            adv_mean = valid_advantages.mean()
            adv_std = valid_advantages.std(unbiased=False).clamp_min(1e-8)
            advantages = (advantages - adv_mean) / adv_std

        batch_size, seq_len = actions.shape
        if obs.dim() == 5:
            obs_flat = obs.reshape(batch_size * seq_len, *obs.shape[2:])
        else:
            obs_flat = obs.reshape(batch_size * seq_len, -1)

        actions_flat = actions.reshape(batch_size * seq_len)
        old_log_probs_flat = old_log_probs.reshape(batch_size * seq_len)
        old_values_flat = old_values.reshape(batch_size * seq_len)
        returns_flat = returns.reshape(batch_size * seq_len)
        advantages_flat = advantages.reshape(batch_size * seq_len)
        mask_flat = action_masks.reshape(batch_size * seq_len)

        prev_actions_flat = None
        prev_rewards_flat = None
        if prev_actions is not None:
            prev_actions_flat = prev_actions.reshape(batch_size * seq_len, -1)
        if prev_rewards is not None:
            prev_rewards_flat = prev_rewards.reshape(batch_size * seq_len, -1)

        valid_indices = torch.nonzero(mask_flat > 0.0, as_tuple=False).squeeze(-1)
        if valid_indices.numel() == 0:
            return {
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy': 0.0,
                'clipfrac': 0.0,
                'mean_return': 0.0,
            }

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_clipfrac = 0.0
        n_updates = 0

        for _ in range(self.ppo_epochs):
            shuffled = valid_indices[torch.randperm(valid_indices.numel(), device=self.device)]

            for start in range(0, shuffled.numel(), self.batch_size):
                idx = shuffled[start:start + self.batch_size]
                if idx.numel() == 0:
                    continue

                log_probs, values, entropy = self.evaluate_actions(
                    obs_flat[idx],
                    actions_flat[idx],
                    prev_actions=None if prev_actions_flat is None else prev_actions_flat[idx],
                    prev_rewards=None if prev_rewards_flat is None else prev_rewards_flat[idx],
                )

                ratio = torch.exp(log_probs - old_log_probs_flat[idx])
                surr_1 = ratio * advantages_flat[idx]
                surr_2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_flat[idx]
                policy_loss = -torch.min(surr_1, surr_2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = old_values_flat[idx] + torch.clamp(
                        values - old_values_flat[idx],
                        -self.clip_epsilon,
                        self.clip_epsilon,
                    )
                    value_loss_unclipped = (values - returns_flat[idx]).pow(2)
                    value_loss_clipped = (value_pred_clipped - returns_flat[idx]).pow(2)
                    value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()
                else:
                    value_loss = (values - returns_flat[idx]).pow(2).mean()

                entropy_bonus = entropy.mean()
                total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_bonus

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    clipfrac = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()

                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy += float(entropy_bonus.item())
                total_clipfrac += clipfrac
                n_updates += 1

        denom = max(n_updates, 1)
        return {
            'policy_loss': total_policy_loss / denom,
            'value_loss': total_value_loss / denom,
            'entropy': total_entropy / denom,
            'clipfrac': total_clipfrac / denom,
            'mean_return': float(returns_flat[valid_indices].mean().item()),
        }

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
