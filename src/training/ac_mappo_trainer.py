"""
MAPPO Trainer for Multi-Agent Almgren-Chriss Optimal Execution.

Combines:
- MAPPOAgent with centralized critic
- LSTMInfoState for history encoding in partial observability
- DiscretizeActionWrapper for continuous-to-discrete action conversion
- Evaluation against Nash equilibrium and TWAP baselines

Author: Yunian Pan
Email: yp1170@nyu.edu
"""

from typing import Dict, Any, Optional, List, Callable
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from tqdm import tqdm
import os
import sys

# Add paths
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.fin.simulations import (
    make_multi_agent_ac_env,
    DiscretizeActionWrapper,
    MultiAgentACParams,
)
from lib.fin.agents import (
    create_nash_multi_agent_policy,
    create_cooperative_multi_agent_policy,
    MultiAgentPolicy,
)

from src.info_states import LSTMInfoState, MLPInfoState
from src.pg_agents import PolicyNetwork, ValueNetwork


@dataclass
class ACMAPPOConfig:
    """Configuration for A-C MAPPO training."""
    # Environment
    n_agents: int = 2
    n_steps: int = 20
    X_0: float = 100_000
    gamma_impact: float = 2.5e-7  # Permanent impact
    eta: float = 2.5e-6           # Temporary impact
    sigma: float = 0.02           # Volatility
    lambda_var: float = 1e-6      # Risk aversion

    # Action discretization
    n_action_bins: int = 21       # 0%, 5%, ..., 100%

    # Info state
    info_state_type: str = 'lstm'  # 'lstm' or 'mlp'
    info_state_dim: int = 64
    lstm_embed_dim: int = 128
    lstm_num_layers: int = 1

    # MAPPO hyperparameters
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    n_ppo_epochs: int = 4
    minibatch_size: int = 64

    # Policy/Value network
    hidden_dim: int = 128
    share_policy: bool = True     # Share policy across agents (symmetric game)
    share_value: bool = True      # Share value function
    centralized_critic: str = 'concat'  # 'concat' or 'global_state'

    # Training
    n_episodes: int = 5000
    update_freq: int = 10         # Update every N episodes
    eval_freq: int = 100
    save_freq: int = 500
    save_path: str = './checkpoints/ac_mappo'
    device: str = 'cpu'
    seed: int = 42


class ACMAPPOTrainer:
    """
    MAPPO trainer specialized for Almgren-Chriss multi-agent execution.

    Features:
    - Centralized training, decentralized execution (CTDE)
    - Info state encoder for partial observability
    - Comparison against Nash equilibrium and TWAP baselines
    """

    def __init__(self, config: ACMAPPOConfig):
        """Initialize trainer with configuration."""
        self.config = config
        self.device = config.device

        # Set seed
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        # Create environment
        self.base_env = make_multi_agent_ac_env(
            n_agents=config.n_agents,
            n_steps=config.n_steps,
            X_0=config.X_0,
            gamma=config.gamma_impact,
            eta=config.eta,
            sigma=config.sigma,
        )
        self.env = DiscretizeActionWrapper(self.base_env, n_bins=config.n_action_bins)

        # Get environment info
        self.agent_ids = list(self.env.possible_agents)
        self.n_agents = len(self.agent_ids)
        sample_agent = self.agent_ids[0]
        self.obs_dim = np.prod(self.env.observation_space(sample_agent).shape)
        self.n_actions = self.env.action_space(sample_agent).n

        print(f"Environment: {self.n_agents} agents, obs_dim={self.obs_dim}, n_actions={self.n_actions}")

        # Create info state encoders
        self._create_networks()

        # Create optimizer
        all_params = []
        seen_ids = set()
        for net in [*self.info_encoders.values(), *self.policies.values(), *self.critics.values()]:
            if id(net) not in seen_ids:
                all_params.extend(list(net.parameters()))
                seen_ids.add(id(net))

        self.optimizer = torch.optim.Adam(all_params, lr=config.lr)

        # Training state
        self.episode_rewards = []
        self.training_metrics = []
        self.hidden_states = {agent_id: None for agent_id in self.agent_ids}
        self.prev_actions = {agent_id: None for agent_id in self.agent_ids}
        self.prev_rewards = {agent_id: None for agent_id in self.agent_ids}

        # Buffer for collecting episodes
        self.episode_buffer = []

    def _create_networks(self):
        """Create info state encoders, policies, and critics."""
        config = self.config

        # Info state encoder factory
        def make_encoder():
            if config.info_state_type == 'lstm':
                return LSTMInfoState(
                    obs_dim=int(self.obs_dim),
                    action_dim=self.n_actions,
                    state_dim=config.info_state_dim,
                    embed_dim=config.lstm_embed_dim,
                    num_layers=config.lstm_num_layers,
                ).to(self.device)
            else:  # mlp
                return MLPInfoState(
                    obs_dim=int(self.obs_dim),
                    action_dim=self.n_actions,
                    state_dim=config.info_state_dim,
                ).to(self.device)

        # Create encoders (shared or independent)
        if config.share_policy:
            shared_encoder = make_encoder()
            self.info_encoders = {agent_id: shared_encoder for agent_id in self.agent_ids}
        else:
            self.info_encoders = {agent_id: make_encoder() for agent_id in self.agent_ids}

        # Create policies (shared or independent)
        def make_policy():
            return PolicyNetwork(
                state_dim=config.info_state_dim,
                action_dim=self.n_actions,
                hidden_dims=[config.hidden_dim, config.hidden_dim],
            ).to(self.device)

        if config.share_policy:
            shared_policy = make_policy()
            self.policies = {agent_id: shared_policy for agent_id in self.agent_ids}
        else:
            self.policies = {agent_id: make_policy() for agent_id in self.agent_ids}

        # Create centralized critic
        if config.centralized_critic == 'concat':
            critic_input_dim = config.info_state_dim * self.n_agents
        else:
            critic_input_dim = config.info_state_dim

        def make_critic():
            return ValueNetwork(
                state_dim=critic_input_dim,
                hidden_dims=[config.hidden_dim, config.hidden_dim],
            ).to(self.device)

        if config.share_value:
            shared_critic = make_critic()
            self.critics = {agent_id: shared_critic for agent_id in self.agent_ids}
        else:
            self.critics = {agent_id: make_critic() for agent_id in self.agent_ids}

    def reset_episode_state(self):
        """Reset hidden states and previous action/reward for new episode."""
        for agent_id in self.agent_ids:
            self.hidden_states[agent_id] = self.info_encoders[agent_id].init_hidden(
                1, self.device
            )
            self.prev_actions[agent_id] = torch.zeros(1, self.n_actions, device=self.device)
            self.prev_rewards[agent_id] = torch.zeros(1, 1, device=self.device)

    def select_actions(
        self,
        observations: Dict[str, np.ndarray],
        deterministic: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Select actions for all agents.

        Returns dict with 'actions' and 'infos' for each agent.
        """
        info_states = {}
        results = {}

        with torch.no_grad():
            # Get info states for all agents
            for agent_id in self.agent_ids:
                obs_tensor = torch.as_tensor(
                    observations[agent_id], dtype=torch.float32, device=self.device
                ).unsqueeze(0)

                info_state, self.hidden_states[agent_id] = self.info_encoders[agent_id](
                    obs_tensor,
                    action=self.prev_actions[agent_id],
                    reward=self.prev_rewards[agent_id],
                    hidden=self.hidden_states[agent_id],
                )
                info_states[agent_id] = info_state

            # Centralized critic input
            if self.config.centralized_critic == 'concat':
                critic_input = torch.cat(
                    [info_states[agent_id] for agent_id in self.agent_ids],
                    dim=-1
                )
            else:
                critic_input = info_states[self.agent_ids[0]]  # Use first agent's

            # Select actions for each agent
            for agent_id in self.agent_ids:
                action, log_prob = self.policies[agent_id].get_action(
                    info_states[agent_id], deterministic=deterministic
                )
                value = self.critics[agent_id](critic_input).squeeze(-1)

                results[agent_id] = {
                    'action': action.item(),
                    'log_prob': log_prob.item(),
                    'value': value.item(),
                    'info_state': info_states[agent_id].cpu().numpy(),
                }

        return results

    def collect_episode(self, seed: Optional[int] = None) -> Dict[str, float]:
        """Collect one episode of experience."""
        obs, infos = self.env.reset(seed=seed)
        self.reset_episode_state()

        episode_data = {
            agent_id: {
                'obs': [],
                'info_states': [],
                'actions': [],
                'log_probs': [],
                'values': [],
                'rewards': [],
                'dones': [],
            }
            for agent_id in self.agent_ids
        }

        episode_rewards = {agent_id: 0.0 for agent_id in self.agent_ids}
        done = False
        step = 0

        while not done and step < self.config.n_steps:
            # Select actions
            action_results = self.select_actions(obs)
            actions = {agent_id: result['action'] for agent_id, result in action_results.items()}

            # Step environment
            next_obs, rewards, terms, truncs, next_infos = self.env.step(actions)

            # Store experience
            for agent_id in self.agent_ids:
                episode_data[agent_id]['obs'].append(obs[agent_id])
                episode_data[agent_id]['info_states'].append(action_results[agent_id]['info_state'])
                episode_data[agent_id]['actions'].append(actions[agent_id])
                episode_data[agent_id]['log_probs'].append(action_results[agent_id]['log_prob'])
                episode_data[agent_id]['values'].append(action_results[agent_id]['value'])
                episode_data[agent_id]['rewards'].append(rewards.get(agent_id, 0.0))
                episode_data[agent_id]['dones'].append(
                    float(terms.get(agent_id, False) or truncs.get(agent_id, False))
                )

                episode_rewards[agent_id] += rewards.get(agent_id, 0.0)

                # Update prev action/reward
                one_hot = torch.zeros(1, self.n_actions, device=self.device)
                one_hot[0, actions[agent_id]] = 1.0
                self.prev_actions[agent_id] = one_hot
                self.prev_rewards[agent_id] = torch.tensor(
                    [[rewards.get(agent_id, 0.0)]], dtype=torch.float32, device=self.device
                )

            obs = next_obs
            done = all(terms.get(a, False) or truncs.get(a, False) for a in self.env.agents)
            step += 1

        # Store episode
        self.episode_buffer.append(episode_data)

        return {
            **{f'{agent_id}_return': episode_rewards[agent_id] for agent_id in self.agent_ids},
            'episode_length': step,
        }

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        next_value: float = 0.0,
    ) -> tuple:
        """Compute GAE advantages and returns."""
        gamma = self.config.gamma
        gae_lambda = self.config.gae_lambda
        T = len(rewards)

        advantages = np.zeros(T)
        gae = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def update(self) -> Dict[str, float]:
        """Update policy and value networks using collected episodes."""
        if not self.episode_buffer:
            return {}

        # Combine all episodes into training data
        all_data = {agent_id: {k: [] for k in ['obs', 'info_states', 'actions', 'log_probs',
                                                'values', 'rewards', 'dones', 'advantages', 'returns']}
                    for agent_id in self.agent_ids}

        for episode_data in self.episode_buffer:
            for agent_id in self.agent_ids:
                ep = episode_data[agent_id]
                rewards = np.array(ep['rewards'])
                values = np.array(ep['values'])
                dones = np.array(ep['dones'])

                advantages, returns = self.compute_gae(rewards, values, dones)

                all_data[agent_id]['obs'].extend(ep['obs'])
                all_data[agent_id]['info_states'].extend(ep['info_states'])
                all_data[agent_id]['actions'].extend(ep['actions'])
                all_data[agent_id]['log_probs'].extend(ep['log_probs'])
                all_data[agent_id]['values'].extend(ep['values'])
                all_data[agent_id]['rewards'].extend(ep['rewards'])
                all_data[agent_id]['dones'].extend(ep['dones'])
                all_data[agent_id]['advantages'].extend(advantages)
                all_data[agent_id]['returns'].extend(returns)

        # Convert to tensors
        for agent_id in self.agent_ids:
            for k in all_data[agent_id]:
                all_data[agent_id][k] = torch.as_tensor(
                    np.array(all_data[agent_id][k]),
                    dtype=torch.float32 if k != 'actions' else torch.long,
                    device=self.device
                )

        # Normalize advantages
        if self.config.share_policy:
            all_advs = torch.cat([all_data[a]['advantages'] for a in self.agent_ids])
            adv_mean, adv_std = all_advs.mean(), all_advs.std() + 1e-8
            for agent_id in self.agent_ids:
                all_data[agent_id]['advantages'] = (all_data[agent_id]['advantages'] - adv_mean) / adv_std
        else:
            for agent_id in self.agent_ids:
                advs = all_data[agent_id]['advantages']
                all_data[agent_id]['advantages'] = (advs - advs.mean()) / (advs.std() + 1e-8)

        # PPO epochs
        n_samples = len(all_data[self.agent_ids[0]]['obs'])
        batch_size = min(self.config.minibatch_size, n_samples)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for epoch in range(self.config.n_ppo_epochs):
            indices = np.random.permutation(n_samples)

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_idx = indices[start:end]

                policy_loss = 0.0
                value_loss = 0.0
                entropy_loss = 0.0

                # Get batch data
                batch_info_states = {
                    agent_id: all_data[agent_id]['info_states'][batch_idx]
                    for agent_id in self.agent_ids
                }

                # Centralized critic input
                if self.config.centralized_critic == 'concat':
                    critic_input = torch.cat(
                        [batch_info_states[agent_id].reshape(len(batch_idx), -1)
                         for agent_id in self.agent_ids],
                        dim=-1
                    )
                else:
                    critic_input = batch_info_states[self.agent_ids[0]].reshape(len(batch_idx), -1)

                for agent_id in self.agent_ids:
                    info_state = batch_info_states[agent_id].reshape(len(batch_idx), -1)
                    actions = all_data[agent_id]['actions'][batch_idx]
                    old_log_probs = all_data[agent_id]['log_probs'][batch_idx]
                    advantages = all_data[agent_id]['advantages'][batch_idx]
                    returns = all_data[agent_id]['returns'][batch_idx]
                    old_values = all_data[agent_id]['values'][batch_idx]

                    # Evaluate actions
                    log_probs, entropy = self.policies[agent_id].evaluate_actions(info_state, actions)
                    values = self.critics[agent_id](critic_input).squeeze(-1)

                    # Policy loss (PPO clipping)
                    ratio = torch.exp(log_probs - old_log_probs)
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon,
                                        1 + self.config.clip_epsilon) * advantages
                    policy_loss += -torch.min(surr1, surr2).mean()

                    # Value loss (clipped)
                    value_pred_clipped = old_values + torch.clamp(
                        values - old_values,
                        -self.config.clip_epsilon,
                        self.config.clip_epsilon
                    )
                    value_loss_unclipped = (values - returns).pow(2)
                    value_loss_clipped = (value_pred_clipped - returns).pow(2)
                    value_loss += torch.max(value_loss_unclipped, value_loss_clipped).mean()

                    # Entropy
                    entropy_loss += -entropy.mean()

                # Average across agents
                policy_loss = policy_loss / self.n_agents
                value_loss = value_loss / self.n_agents
                entropy_loss = entropy_loss / self.n_agents

                # Total loss
                loss = (policy_loss +
                        self.config.value_coef * value_loss +
                        self.config.entropy_coef * entropy_loss)

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for group in self.optimizer.param_groups for p in group['params']],
                    self.config.max_grad_norm
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += -entropy_loss.item()
                n_updates += 1

        # Clear buffer
        self.episode_buffer = []

        return {
            'policy_loss': total_policy_loss / n_updates if n_updates > 0 else 0,
            'value_loss': total_value_loss / n_updates if n_updates > 0 else 0,
            'entropy': total_entropy / n_updates if n_updates > 0 else 0,
        }

    def evaluate(self, n_episodes: int = 10) -> Dict[str, float]:
        """Evaluate current policy."""
        eval_rewards = {agent_id: [] for agent_id in self.agent_ids}

        for ep in range(n_episodes):
            obs, _ = self.env.reset(seed=10000 + ep)
            self.reset_episode_state()

            episode_rewards = {agent_id: 0.0 for agent_id in self.agent_ids}
            done = False
            step = 0

            while not done and step < self.config.n_steps:
                action_results = self.select_actions(obs, deterministic=True)
                actions = {agent_id: result['action'] for agent_id, result in action_results.items()}

                obs, rewards, terms, truncs, _ = self.env.step(actions)

                for agent_id in self.agent_ids:
                    episode_rewards[agent_id] += rewards.get(agent_id, 0.0)

                    one_hot = torch.zeros(1, self.n_actions, device=self.device)
                    one_hot[0, actions[agent_id]] = 1.0
                    self.prev_actions[agent_id] = one_hot
                    self.prev_rewards[agent_id] = torch.tensor(
                        [[rewards.get(agent_id, 0.0)]], dtype=torch.float32, device=self.device
                    )

                done = all(terms.get(a, False) or truncs.get(a, False) for a in self.env.agents)
                step += 1

            for agent_id in self.agent_ids:
                eval_rewards[agent_id].append(episode_rewards[agent_id])

        stats = {}
        for agent_id in self.agent_ids:
            stats[f'{agent_id}_mean'] = np.mean(eval_rewards[agent_id])
            stats[f'{agent_id}_std'] = np.std(eval_rewards[agent_id])

        return stats

    def evaluate_vs_baselines(self, n_episodes: int = 50) -> Dict[str, float]:
        """Compare learned policy against Nash and TWAP baselines."""
        results = {}

        # Learned policy
        learned_stats = self.evaluate(n_episodes)
        results['learned_mean'] = np.mean([learned_stats[f'{a}_mean'] for a in self.agent_ids])

        # Nash equilibrium
        nash_rewards = []
        nash_policies = create_nash_multi_agent_policy(self.base_env)
        nash_policy = MultiAgentPolicy(nash_policies)

        for ep in range(n_episodes):
            obs, infos = self.base_env.reset(seed=20000 + ep)
            total_reward = 0.0
            done = False

            while not done:
                actions = nash_policy.act(obs, infos)
                obs, rewards, terms, truncs, infos = self.base_env.step(actions)
                total_reward += sum(rewards.values())
                done = all(truncs.values())

            nash_rewards.append(total_reward / self.n_agents)
            nash_policy.reset()

        results['nash_mean'] = np.mean(nash_rewards)
        results['nash_std'] = np.std(nash_rewards)

        # TWAP
        twap_rewards = []
        twap_policy = MultiAgentPolicy.all_twap(self.base_env.possible_agents)

        for ep in range(n_episodes):
            obs, infos = self.base_env.reset(seed=30000 + ep)
            total_reward = 0.0
            done = False

            while not done:
                actions = twap_policy.act(obs, infos)
                obs, rewards, terms, truncs, infos = self.base_env.step(actions)
                total_reward += sum(rewards.values())
                done = all(truncs.values())

            twap_rewards.append(total_reward / self.n_agents)
            twap_policy.reset()

        results['twap_mean'] = np.mean(twap_rewards)
        results['twap_std'] = np.std(twap_rewards)

        # Relative performance
        results['vs_nash'] = results['learned_mean'] - results['nash_mean']
        results['vs_twap'] = results['learned_mean'] - results['twap_mean']

        return results

    def train(self, progress_bar: bool = True):
        """Main training loop."""
        print(f"Starting MAPPO training for {self.config.n_episodes} episodes...")
        print(f"Agents: {self.agent_ids}")
        print(f"Device: {self.device}")

        episode_iter = range(self.config.n_episodes)
        if progress_bar:
            episode_iter = tqdm(episode_iter, desc="Training")

        for episode in episode_iter:
            # Collect episode
            episode_stats = self.collect_episode(seed=episode)
            self.episode_rewards.append(episode_stats)

            # Update
            if (episode + 1) % self.config.update_freq == 0:
                metrics = self.update()
                self.training_metrics.append(metrics)

                if progress_bar:
                    recent = [s[f'{self.agent_ids[0]}_return'] for s in self.episode_rewards[-10:]]
                    episode_iter.set_postfix({
                        'avg_return': f'{np.mean(recent):.4f}',
                        'p_loss': f'{metrics.get("policy_loss", 0):.4f}',
                    })

            # Evaluate
            if (episode + 1) % self.config.eval_freq == 0:
                eval_stats = self.evaluate(n_episodes=10)
                if progress_bar:
                    tqdm.write(f"Episode {episode + 1}: {eval_stats}")

            # Save
            if (episode + 1) % self.config.save_freq == 0:
                self.save_checkpoint(episode + 1)

        # Final evaluation vs baselines
        print("\nFinal evaluation vs baselines...")
        baseline_stats = self.evaluate_vs_baselines(n_episodes=50)
        print(f"Learned: {baseline_stats['learned_mean']:.4f}")
        print(f"Nash: {baseline_stats['nash_mean']:.4f} ± {baseline_stats['nash_std']:.4f}")
        print(f"TWAP: {baseline_stats['twap_mean']:.4f} ± {baseline_stats['twap_std']:.4f}")
        print(f"vs Nash: {baseline_stats['vs_nash']:+.4f}")
        print(f"vs TWAP: {baseline_stats['vs_twap']:+.4f}")

        print("\nTraining completed!")
        return self.episode_rewards, self.training_metrics

    def save_checkpoint(self, episode: int):
        """Save checkpoint."""
        os.makedirs(self.config.save_path, exist_ok=True)

        checkpoint = {
            'episode': episode,
            'config': self.config,
            'info_encoders': {},
            'policies': {},
            'critics': {},
            'optimizer': self.optimizer.state_dict(),
        }

        # Save unique modules
        saved_ids = set()
        for i, (agent_id, encoder) in enumerate(self.info_encoders.items()):
            if id(encoder) not in saved_ids:
                checkpoint['info_encoders'][i] = encoder.state_dict()
                saved_ids.add(id(encoder))

        saved_ids = set()
        for i, (agent_id, policy) in enumerate(self.policies.items()):
            if id(policy) not in saved_ids:
                checkpoint['policies'][i] = policy.state_dict()
                saved_ids.add(id(policy))

        saved_ids = set()
        for i, (agent_id, critic) in enumerate(self.critics.items()):
            if id(critic) not in saved_ids:
                checkpoint['critics'][i] = critic.state_dict()
                saved_ids.add(id(critic))

        path = f"{self.config.save_path}/checkpoint_ep{episode}.pt"
        torch.save(checkpoint, path)
        print(f"\nCheckpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        # Load modules
        loaded_ids = {}
        for i, state_dict in checkpoint['info_encoders'].items():
            encoder = list(self.info_encoders.values())[i]
            if id(encoder) not in loaded_ids:
                encoder.load_state_dict(state_dict)
                loaded_ids[id(encoder)] = True

        loaded_ids = {}
        for i, state_dict in checkpoint['policies'].items():
            policy = list(self.policies.values())[i]
            if id(policy) not in loaded_ids:
                policy.load_state_dict(state_dict)
                loaded_ids[id(policy)] = True

        loaded_ids = {}
        for i, state_dict in checkpoint['critics'].items():
            critic = list(self.critics.values())[i]
            if id(critic) not in loaded_ids:
                critic.load_state_dict(state_dict)
                loaded_ids[id(critic)] = True

        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Loaded checkpoint: {path}")
