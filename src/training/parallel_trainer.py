"""
Trainer for PettingZoo ParallelEnv environments.

Supports simultaneous multi-agent action collection for environments
where all agents act at the same time (e.g., Almgren-Chriss execution).

Author: Yunian Pan
Email: yp1170@nyu.edu
"""

from typing import Dict, Any, Optional, List, Callable
import numpy as np
import torch
from tqdm import tqdm
from dataclasses import dataclass, field
import os

from .buffer import MultiAgentRolloutBuffer


@dataclass
class ParallelTrainerConfig:
    """Configuration for ParallelEnv trainer."""
    device: str = 'cpu'
    n_episodes: int = 1000
    max_steps: int = 100
    update_freq: int = 10  # Update every N episodes
    eval_freq: int = 50
    save_freq: int = 100
    save_path: str = './checkpoints'
    gamma: float = 0.99
    gae_lambda: float = 0.95
    normalize_advantages: bool = True


class ParallelEnvTrainer:
    """
    Trainer for PettingZoo ParallelEnv with policy gradient agents.

    Unlike AEC environments where agents take turns, ParallelEnv has
    all agents act simultaneously each step. This trainer collects
    trajectories for all agents in parallel.
    """

    def __init__(
        self,
        env,
        agents: Dict[str, Any],
        config: ParallelTrainerConfig,
    ):
        """
        Args:
            env: PettingZoo ParallelEnv
            agents: Dict mapping agent_id -> Agent (with get_action, update methods)
            config: Training configuration
        """
        self.env = env
        self.agents = agents
        self.config = config
        self.device = config.device

        # Get environment info
        sample_agent = env.possible_agents[0]
        self.obs_shape = env.observation_space(sample_agent).shape
        self.n_actions = env.action_space(sample_agent).n
        self.agent_ids = list(env.possible_agents)

        # Create buffer
        self.buffer = MultiAgentRolloutBuffer(
            buffer_size=config.max_steps * len(self.agent_ids) * config.update_freq,
            obs_shape=self.obs_shape,
            action_dim=self.n_actions,
            agent_ids=self.agent_ids,
            device=self.device,
        )

        # Metrics
        self.episode_rewards = []
        self.training_metrics = []

    def collect_episode(self, seed: Optional[int] = None) -> Dict[str, float]:
        """
        Collect one episode of experience.

        All agents act simultaneously at each step.

        Returns:
            Episode statistics
        """
        obs, infos = self.env.reset(seed=seed)

        # Initialize hidden states and previous action/reward for each agent
        hiddens = {}
        prev_actions = {}
        prev_rewards = {}

        for agent_id in self.agent_ids:
            if hasattr(self.agents[agent_id], 'info_state_encoder'):
                hiddens[agent_id] = self.agents[agent_id].info_state_encoder.init_hidden(
                    1, self.device
                )
            else:
                hiddens[agent_id] = None

            prev_actions[agent_id] = torch.zeros(1, self.n_actions, device=self.device)
            prev_rewards[agent_id] = torch.zeros(1, 1, device=self.device)

        # Episode data storage
        episode_data = {
            agent_id: {
                'obs': [],
                'actions': [],
                'rewards': [],
                'dones': [],
                'log_probs': [],
                'values': [],
                'prev_actions': [],
                'prev_rewards': [],
                'action_masks': [],
            }
            for agent_id in self.agent_ids
        }

        episode_rewards = {agent_id: 0.0 for agent_id in self.agent_ids}
        episode_length = 0

        done = False
        while not done:
            actions = {}
            action_infos = {}

            # All agents select actions simultaneously
            for agent_id in self.env.agents:
                obs_tensor = torch.as_tensor(
                    obs[agent_id], dtype=torch.float32, device=self.device
                ).unsqueeze(0)

                with torch.no_grad():
                    agent = self.agents[agent_id]
                    result = agent.get_action(
                        obs_tensor,
                        prev_action=prev_actions[agent_id],
                        prev_reward=prev_rewards[agent_id],
                        hidden=hiddens[agent_id],
                        deterministic=False,
                    )

                    if len(result) == 4:  # A2C/PPO returns (action, log_prob, value, hidden)
                        action, log_prob, value, hiddens[agent_id] = result
                        value = value.item() if hasattr(value, 'item') else float(value)
                    else:  # REINFORCE returns (action, log_prob, hidden)
                        action, log_prob, hiddens[agent_id] = result
                        value = 0.0

                    actions[agent_id] = action.item()
                    action_infos[agent_id] = {
                        'log_prob': log_prob.item(),
                        'value': value,
                    }

            # Step environment with all actions
            next_obs, rewards, terms, truncs, next_infos = self.env.step(actions)

            # Store experience for each agent
            for agent_id in self.agent_ids:
                episode_data[agent_id]['obs'].append(obs[agent_id])
                episode_data[agent_id]['actions'].append(actions[agent_id])
                episode_data[agent_id]['rewards'].append(rewards.get(agent_id, 0.0))
                episode_data[agent_id]['dones'].append(
                    float(terms.get(agent_id, False) or truncs.get(agent_id, False))
                )
                episode_data[agent_id]['log_probs'].append(action_infos[agent_id]['log_prob'])
                episode_data[agent_id]['values'].append(action_infos[agent_id]['value'])
                episode_data[agent_id]['prev_actions'].append(
                    prev_actions[agent_id].squeeze(0).cpu().numpy()
                )
                episode_data[agent_id]['prev_rewards'].append(
                    prev_rewards[agent_id].squeeze(0).cpu().numpy()
                )
                episode_data[agent_id]['action_masks'].append(1.0)

                episode_rewards[agent_id] += rewards.get(agent_id, 0.0)

                # Update previous action/reward for next step
                one_hot = torch.zeros(1, self.n_actions, device=self.device)
                one_hot[0, actions[agent_id]] = 1.0
                prev_actions[agent_id] = one_hot
                prev_rewards[agent_id] = torch.tensor(
                    [[rewards.get(agent_id, 0.0)]], dtype=torch.float32, device=self.device
                )

            # Update state
            obs = next_obs
            infos = next_infos
            episode_length += 1

            # Check if all agents are done
            done = all(terms.get(a, False) or truncs.get(a, False) for a in self.env.agents)

            # Safety check for max steps
            if episode_length >= self.config.max_steps:
                done = True

        # Add episode to buffer
        for agent_id in self.agent_ids:
            self.buffer.add_episode(agent_id, episode_data[agent_id])
        self.buffer.episode_lengths.append(episode_length)

        return {
            **{f'{agent_id}_return': episode_rewards[agent_id] for agent_id in episode_rewards},
            'episode_length': episode_length,
        }

    def update_agents(self) -> Dict[str, Dict[str, float]]:
        """
        Update all agents from buffer.

        Returns:
            Dict mapping agent_id -> training metrics
        """
        all_metrics = {}

        for agent_id, agent in self.agents.items():
            rollout_data = self.buffer.get(agent_id)

            if len(rollout_data['obs']) == 0:
                continue

            metrics = agent.update(rollout_data)
            all_metrics[agent_id] = metrics

        # Clear buffer after update
        self.buffer.clear()

        return all_metrics

    def evaluate(
        self,
        n_episodes: int = 10,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate agents.

        Args:
            n_episodes: Number of evaluation episodes
            deterministic: Use deterministic actions

        Returns:
            Evaluation statistics
        """
        eval_rewards = {agent_id: [] for agent_id in self.agent_ids}

        for ep in range(n_episodes):
            obs, infos = self.env.reset(seed=1000 + ep)

            # Initialize hidden states
            hiddens = {}
            prev_actions = {}
            prev_rewards = {}

            for agent_id in self.agent_ids:
                if hasattr(self.agents[agent_id], 'info_state_encoder'):
                    hiddens[agent_id] = self.agents[agent_id].info_state_encoder.init_hidden(
                        1, self.device
                    )
                else:
                    hiddens[agent_id] = None
                prev_actions[agent_id] = torch.zeros(1, self.n_actions, device=self.device)
                prev_rewards[agent_id] = torch.zeros(1, 1, device=self.device)

            episode_rewards = {agent_id: 0.0 for agent_id in self.agent_ids}
            done = False
            steps = 0

            while not done and steps < self.config.max_steps:
                actions = {}

                for agent_id in self.env.agents:
                    obs_tensor = torch.as_tensor(
                        obs[agent_id], dtype=torch.float32, device=self.device
                    ).unsqueeze(0)

                    with torch.no_grad():
                        agent = self.agents[agent_id]
                        result = agent.get_action(
                            obs_tensor,
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            hidden=hiddens[agent_id],
                            deterministic=deterministic,
                        )

                        if len(result) == 4:
                            action, _, _, hiddens[agent_id] = result
                        else:
                            action, _, hiddens[agent_id] = result

                        actions[agent_id] = action.item()

                obs, rewards, terms, truncs, infos = self.env.step(actions)

                for agent_id in self.agent_ids:
                    episode_rewards[agent_id] += rewards.get(agent_id, 0.0)

                    one_hot = torch.zeros(1, self.n_actions, device=self.device)
                    one_hot[0, actions[agent_id]] = 1.0
                    prev_actions[agent_id] = one_hot
                    prev_rewards[agent_id] = torch.tensor(
                        [[rewards.get(agent_id, 0.0)]], dtype=torch.float32, device=self.device
                    )

                done = all(terms.get(a, False) or truncs.get(a, False) for a in self.env.agents)
                steps += 1

            for agent_id in episode_rewards:
                eval_rewards[agent_id].append(episode_rewards[agent_id])

        # Compute statistics
        stats = {}
        for agent_id in eval_rewards:
            stats[f'{agent_id}_mean'] = np.mean(eval_rewards[agent_id])
            stats[f'{agent_id}_std'] = np.std(eval_rewards[agent_id])

        return stats

    def train(self, progress_bar: bool = True) -> tuple:
        """
        Main training loop.

        Args:
            progress_bar: Show progress bar

        Returns:
            (episode_rewards, training_metrics)
        """
        print(f"Starting training for {self.config.n_episodes} episodes...")
        print(f"Agents: {list(self.agents.keys())}")
        print(f"Update frequency: {self.config.update_freq} episodes")

        episode_iter = range(self.config.n_episodes)
        if progress_bar:
            episode_iter = tqdm(episode_iter, desc="Training")

        for episode in episode_iter:
            # Collect episode
            episode_stats = self.collect_episode(seed=episode)
            self.episode_rewards.append(episode_stats)

            # Update agents
            if (episode + 1) % self.config.update_freq == 0:
                metrics = self.update_agents()
                self.training_metrics.append(metrics)

                # Update progress bar
                if progress_bar:
                    primary = self.agent_ids[0]
                    recent_returns = [
                        stats[f'{primary}_return']
                        for stats in self.episode_rewards[-10:]
                    ]
                    avg_return = np.mean(recent_returns) if recent_returns else 0
                    episode_iter.set_postfix({
                        'avg_return': f'{avg_return:.3f}',
                        'buffer': len(self.buffer),
                    })

            # Evaluate
            if (episode + 1) % self.config.eval_freq == 0:
                eval_stats = self.evaluate(n_episodes=10)
                if progress_bar:
                    tqdm.write(f"Episode {episode + 1} - Eval: {eval_stats}")

            # Save checkpoint
            if (episode + 1) % self.config.save_freq == 0:
                self.save_checkpoint(episode + 1)

        print("\nTraining completed!")
        return self.episode_rewards, self.training_metrics

    def save_checkpoint(self, episode: int):
        """Save agent checkpoints."""
        os.makedirs(self.config.save_path, exist_ok=True)

        for agent_id, agent in self.agents.items():
            path = f"{self.config.save_path}/{agent_id}_ep{episode}.pt"
            torch.save({
                'episode': episode,
                'agent_state_dict': agent.state_dict(),
                'config': self.config,
            }, path)

        print(f"\nCheckpoint saved at episode {episode}")

    def load_checkpoint(self, episode: int):
        """Load agent checkpoints."""
        for agent_id, agent in self.agents.items():
            path = f"{self.config.save_path}/{agent_id}_ep{episode}.pt"
            checkpoint = torch.load(path, map_location=self.device)
            agent.load_state_dict(checkpoint['agent_state_dict'])

        print(f"Loaded checkpoint from episode {episode}")
