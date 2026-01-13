"""Multi-Agent Proximal Policy Optimization (MAPPO)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .ppo import PPOAgent
from .networks import PolicyNetwork, ValueNetwork


class MAPPOAgent:
    """MAPPO with centralized critic and decentralized actors.

    Key features:
    - Centralized critic: sees global state or all agents' info states
    - Decentralized actors: each agent acts based on local observations
    - Shared or independent parameters across agents
    """

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
            obs_space: Observation space (per agent)
            action_space: Action space (per agent)
            num_agents: Number of agents
            info_state_encoder_fn: Function that creates info state encoder
            config: Configuration dict with:
                - share_policy: Share policy across agents (default: False)
                - share_value: Share value function across agents (default: True)
                - centralized_critic_input: 'concat' or 'global_state' (default: 'concat')
                - lr: Learning rate
                - gamma: Discount factor
                - gae_lambda: GAE lambda
                - clip_epsilon: PPO clip parameter
                - value_coef: Value loss coefficient
                - entropy_coef: Entropy coefficient
                - max_grad_norm: Gradient clipping
                - device: torch device
        """
        self.num_agents = num_agents
        self.obs_space = obs_space
        self.action_space = action_space
        self.config = config
        self.device = config.get('device', 'cpu')

        # Configuration
        self.share_policy = config.get('share_policy', False)
        self.share_value = config.get('share_value', True)
        self.centralized_critic_input = config.get('centralized_critic_input', 'concat')

        # PPO hyperparameters
        self.lr = config.get('lr', 3e-4)
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        self.value_coef = config.get('value_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.normalize_advantages = config.get('normalize_advantages', True)
        self.use_clipped_value_loss = config.get('use_clipped_value_loss', True)

        # Create info state encoders (one per agent or shared)
        if self.share_policy:
            # Single shared encoder
            shared_encoder = info_state_encoder_fn()
            self.info_state_encoders = {i: shared_encoder for i in range(num_agents)}
        else:
            # Independent encoders
            self.info_state_encoders = {
                i: info_state_encoder_fn() for i in range(num_agents)
            }

        state_dim = list(self.info_state_encoders.values())[0].state_dim
        n_actions = action_space.n

        # Create policies (one per agent or shared)
        if self.share_policy:
            shared_policy = PolicyNetwork(
                state_dim=state_dim,
                n_actions=n_actions,
                hidden_dim=config.get('hidden_dim', 128),
            ).to(self.device)
            self.policies = {i: shared_policy for i in range(num_agents)}
        else:
            self.policies = {
                i: PolicyNetwork(
                    state_dim=state_dim,
                    n_actions=n_actions,
                    hidden_dim=config.get('hidden_dim', 128),
                ).to(self.device)
                for i in range(num_agents)
            }

        # Create centralized critic
        if self.centralized_critic_input == 'concat':
            # Critic sees concatenation of all agents' info states
            critic_input_dim = state_dim * num_agents
        elif self.centralized_critic_input == 'global_state':
            # Critic sees global state (needs to be provided)
            critic_input_dim = config.get('global_state_dim', state_dim * num_agents)
        else:
            raise ValueError(f"Unknown centralized_critic_input: {self.centralized_critic_input}")

        if self.share_value:
            # Single shared critic
            shared_critic = ValueNetwork(
                state_dim=critic_input_dim,
                hidden_dim=config.get('hidden_dim', 128),
            ).to(self.device)
            self.critics = {i: shared_critic for i in range(num_agents)}
        else:
            # Independent critics (still centralized, but separate per agent)
            self.critics = {
                i: ValueNetwork(
                    state_dim=critic_input_dim,
                    hidden_dim=config.get('hidden_dim', 128),
                ).to(self.device)
                for i in range(num_agents)
            }

        # Collect unique parameters
        unique_params = []
        seen_ids = set()

        for encoder in self.info_state_encoders.values():
            if id(encoder) not in seen_ids:
                unique_params.extend(list(encoder.parameters()))
                seen_ids.add(id(encoder))

        for policy in self.policies.values():
            if id(policy) not in seen_ids:
                unique_params.extend(list(policy.parameters()))
                seen_ids.add(id(policy))

        for critic in self.critics.values():
            if id(critic) not in seen_ids:
                unique_params.extend(list(critic.parameters()))
                seen_ids.add(id(critic))

        self.optimizer = torch.optim.Adam(unique_params, lr=self.lr)

        # Hidden states for each agent
        self.hidden_states = {i: None for i in range(num_agents)}

    def select_actions(
        self,
        observations: Dict[int, torch.Tensor],
        global_state: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        actions_prev: Optional[Dict[int, torch.Tensor]] = None,
        rewards_prev: Optional[Dict[int, torch.Tensor]] = None,
    ) -> Tuple[Dict[int, int], Dict[int, Dict[str, Any]]]:
        """Select actions for all agents.

        Args:
            observations: Dict {agent_id: observation}
            global_state: Optional global state for centralized critic
            deterministic: If True, select argmax actions
            actions_prev: Previous actions per agent
            rewards_prev: Previous rewards per agent

        Returns:
            actions: Dict {agent_id: action}
            infos: Dict {agent_id: info_dict}
        """
        with torch.no_grad():
            actions = {}
            infos = {}
            info_states = {}

            # Get info states for all agents
            for i, obs in observations.items():
                action_prev = actions_prev[i] if actions_prev else None
                reward_prev = rewards_prev[i] if rewards_prev else None

                info_state, self.hidden_states[i] = self.info_state_encoders[i](
                    obs, action_prev, reward_prev, self.hidden_states[i]
                )
                info_states[i] = info_state

            # Centralized critic input
            if self.centralized_critic_input == 'concat':
                # Concatenate all info states
                critic_input = torch.cat([info_states[i] for i in range(self.num_agents)], dim=-1)
            elif self.centralized_critic_input == 'global_state':
                critic_input = global_state
            else:
                raise ValueError(f"Unknown centralized_critic_input: {self.centralized_critic_input}")

            # Select actions for each agent
            for i in range(self.num_agents):
                # Policy (decentralized)
                action_logits = self.policies[i](info_states[i])
                probs = F.softmax(action_logits, dim=-1)
                dist = Categorical(probs)

                if deterministic:
                    action = torch.argmax(probs, dim=-1)
                else:
                    action = dist.sample()

                # Value (centralized)
                value = self.critics[i](critic_input)

                actions[i] = action.item()
                infos[i] = {
                    'log_prob': dist.log_prob(action).item(),
                    'value': value.item(),
                    'entropy': dist.entropy().item(),
                    'info_state': info_states[i].detach(),
                    'hidden': self.hidden_states[i],
                }

            return actions, infos

    def evaluate_actions(
        self,
        observations: Dict[int, torch.Tensor],
        actions: Dict[int, torch.Tensor],
        global_states: Optional[torch.Tensor] = None,
        actions_prev: Optional[Dict[int, torch.Tensor]] = None,
        rewards_prev: Optional[Dict[int, torch.Tensor]] = None,
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Evaluate actions for all agents.

        Returns:
            Dict {agent_id: (log_probs, values, entropy)}
        """
        info_states = {}

        # Get info states for all agents
        for i in range(self.num_agents):
            action_prev = actions_prev[i] if actions_prev else None
            reward_prev = rewards_prev[i] if rewards_prev else None

            info_state, _ = self.info_state_encoders[i](
                observations[i], action_prev, reward_prev, None
            )
            info_states[i] = info_state

        # Centralized critic input
        if self.centralized_critic_input == 'concat':
            # Concatenate all info states
            critic_input = torch.cat([info_states[i] for i in range(self.num_agents)], dim=-1)
        elif self.centralized_critic_input == 'global_state':
            critic_input = global_states
        else:
            raise ValueError(f"Unknown centralized_critic_input: {self.centralized_critic_input}")

        results = {}
        for i in range(self.num_agents):
            # Policy
            action_logits = self.policies[i](info_states[i])
            probs = F.softmax(action_logits, dim=-1)
            dist = Categorical(probs)

            log_probs = dist.log_prob(actions[i])
            entropy = dist.entropy()

            # Centralized value
            values = self.critics[i](critic_input).squeeze(-1)

            results[i] = (log_probs, values, entropy)

        return results

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE (same as PPO)."""
        T = len(rewards)
        advantages = torch.zeros(T, device=self.device)
        gae = 0

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
        rollout_buffers: Dict[int, Any],
        n_epochs: int = 10,
        batch_size: int = 64,
    ) -> Dict[str, float]:
        """Update all agents using MAPPO.

        Args:
            rollout_buffers: Dict {agent_id: rollout_buffer}
            n_epochs: Number of epochs
            batch_size: Minibatch size

        Returns:
            Dict with training statistics
        """
        # Compute advantages for each agent
        for agent_id, buffer in rollout_buffers.items():
            advantages, returns = self.compute_gae(
                buffer.rewards,
                buffer.values,
                buffer.dones,
                buffer.next_value,
            )

            if self.normalize_advantages:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            buffer.advantages = advantages
            buffer.returns = returns

        # Training statistics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_clipfrac = 0
        n_updates = 0

        # Training epochs
        for epoch in range(n_epochs):
            # Assuming all buffers have same length
            buffer_len = len(rollout_buffers[0])
            indices = np.arange(buffer_len)
            np.random.shuffle(indices)

            for start in range(0, buffer_len, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]

                # Prepare batch data for all agents
                obs_batch = {}
                actions_batch = {}
                old_log_probs_batch = {}
                advantages_batch = {}
                returns_batch = {}
                old_values_batch = {}
                global_states_batch = None

                for agent_id, buffer in rollout_buffers.items():
                    obs_batch[agent_id] = buffer.observations[batch_indices]
                    actions_batch[agent_id] = buffer.actions[batch_indices]
                    old_log_probs_batch[agent_id] = buffer.log_probs[batch_indices]
                    advantages_batch[agent_id] = buffer.advantages[batch_indices]
                    returns_batch[agent_id] = buffer.returns[batch_indices]
                    old_values_batch[agent_id] = buffer.values[batch_indices]

                    if hasattr(buffer, 'global_states') and buffer.global_states is not None:
                        global_states_batch = buffer.global_states[batch_indices]

                # Evaluate current policy
                eval_results = self.evaluate_actions(
                    obs_batch, actions_batch, global_states_batch
                )

                # Compute losses for all agents
                policy_loss = 0
                value_loss = 0
                entropy_loss = 0
                clipfrac_total = 0

                for agent_id in range(self.num_agents):
                    log_probs, values, entropy = eval_results[agent_id]

                    # PPO policy loss
                    ratio = torch.exp(log_probs - old_log_probs_batch[agent_id])
                    surr1 = ratio * advantages_batch[agent_id]
                    surr2 = torch.clamp(
                        ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon
                    ) * advantages_batch[agent_id]
                    policy_loss += -torch.min(surr1, surr2).mean()

                    # Value loss
                    if self.use_clipped_value_loss:
                        value_pred_clipped = old_values_batch[agent_id] + torch.clamp(
                            values - old_values_batch[agent_id],
                            -self.clip_epsilon,
                            self.clip_epsilon,
                        )
                        value_loss_unclipped = F.mse_loss(values, returns_batch[agent_id])
                        value_loss_clipped = F.mse_loss(value_pred_clipped, returns_batch[agent_id])
                        value_loss += torch.max(value_loss_unclipped, value_loss_clipped)
                    else:
                        value_loss += F.mse_loss(values, returns_batch[agent_id])

                    # Entropy
                    entropy_loss += -entropy.mean()

                    # Clipfrac
                    with torch.no_grad():
                        clipfrac = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean()
                        clipfrac_total += clipfrac.item()

                # Average losses across agents
                policy_loss = policy_loss / self.num_agents
                value_loss = value_loss / self.num_agents
                entropy_loss = entropy_loss / self.num_agents

                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.optimizer.param_groups[0]['params'],
                    self.max_grad_norm,
                )
                self.optimizer.step()

                # Track statistics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += -entropy_loss.item()
                total_clipfrac += clipfrac_total / self.num_agents
                n_updates += 1

        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates,
            'clipfrac': total_clipfrac / n_updates,
        }

    def reset_hidden(self):
        """Reset hidden states for all agents."""
        for i in range(self.num_agents):
            self.hidden_states[i] = None

    def save(self, path: str):
        """Save checkpoint."""
        checkpoint = {}

        # Save unique modules
        saved_ids = set()
        for i, encoder in enumerate(self.info_state_encoders.values()):
            if id(encoder) not in saved_ids:
                checkpoint[f'info_state_encoder_{i}'] = encoder.state_dict()
                saved_ids.add(id(encoder))

        saved_ids = set()
        for i, policy in enumerate(self.policies.values()):
            if id(policy) not in saved_ids:
                checkpoint[f'policy_{i}'] = policy.state_dict()
                saved_ids.add(id(policy))

        saved_ids = set()
        for i, critic in enumerate(self.critics.values()):
            if id(critic) not in saved_ids:
                checkpoint[f'critic_{i}'] = critic.state_dict()
                saved_ids.add(id(critic))

        checkpoint['optimizer'] = self.optimizer.state_dict()
        torch.save(checkpoint, path)

    def load(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        # Load modules (handle shared parameters)
        loaded_ids = {}
        for key in checkpoint:
            if key.startswith('info_state_encoder_'):
                idx = int(key.split('_')[-1])
                encoder = list(self.info_state_encoders.values())[idx]
                if id(encoder) not in loaded_ids:
                    encoder.load_state_dict(checkpoint[key])
                    loaded_ids[id(encoder)] = True

            elif key.startswith('policy_'):
                idx = int(key.split('_')[-1])
                policy = list(self.policies.values())[idx]
                if id(policy) not in loaded_ids:
                    policy.load_state_dict(checkpoint[key])
                    loaded_ids[id(policy)] = True

            elif key.startswith('critic_'):
                idx = int(key.split('_')[-1])
                critic = list(self.critics.values())[idx]
                if id(critic) not in loaded_ids:
                    critic.load_state_dict(checkpoint[key])
                    loaded_ids[id(critic)] = True

        self.optimizer.load_state_dict(checkpoint['optimizer'])
