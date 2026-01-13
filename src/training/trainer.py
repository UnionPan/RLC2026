"""Main training loop for multi-agent RL."""

from typing import Dict, Any, Optional
import numpy as np
import torch
from tqdm import tqdm

from .buffer import MultiAgentRolloutBuffer


class MultiAgentTrainer:
    """Trainer for multi-agent environments."""

    def __init__(
        self,
        env,
        agents: Dict[str, Any],
        config: Dict[str, Any],
    ):
        """
        Args:
            env: PettingZoo AEC environment
            agents: Dictionary {agent_id: Agent}
            config: Training configuration
        """
        self.env = env
        self.agents = agents
        self.config = config

        self.device = config.get('device', 'cpu')
        self.n_episodes = config.get('n_episodes', 1000)
        self.max_steps = config.get('max_steps', 200)
        self.update_freq = config.get('update_freq', 10)  # Update every N episodes
        self.eval_freq = config.get('eval_freq', 50)
        self.save_freq = config.get('save_freq', 100)
        self.save_path = config.get('save_path', './checkpoints')

        # Get environment info
        sample_agent = self.env.possible_agents[0]
        obs_space = self.env.observation_space(sample_agent)
        self.obs_shape = obs_space.shape
        self.num_agents = len(self.env.possible_agents)

        # Create buffer
        self.buffer = MultiAgentRolloutBuffer(
            buffer_size=self.max_steps * self.num_agents * self.update_freq,
            obs_shape=self.obs_shape,
            action_dim=self.env.action_space(sample_agent).n,
            agent_ids=self.env.possible_agents,
            device=self.device,
        )

        # Metrics
        self.episode_rewards = []
        self.training_metrics = []

    def collect_episode(self) -> Dict[str, float]:
        """Collect one episode of experience.

        Returns:
            Episode statistics
        """
        self.env.reset()
        hiddens = {agent_id: self.agents[agent_id].info_state_encoder.init_hidden(1, self.device)
                   for agent_id in self.env.possible_agents}
        prev_actions = {agent_id: torch.zeros(1, self.env.action_space(agent_id).n, device=self.device)
                        for agent_id in self.env.possible_agents}
        prev_rewards = {agent_id: torch.zeros(1, 1, device=self.device)
                        for agent_id in self.env.possible_agents}

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
            for agent_id in self.env.possible_agents
        }

        episode_rewards = {agent_id: 0.0 for agent_id in self.env.possible_agents}
        episode_length = 0

        for agent_id in self.env.agent_iter():
            obs, reward, termination, truncation, info = self.env.last()
            done = termination or truncation

            action_mask = 0.0
            if done:
                action = None
                log_prob = 0.0
                value = 0.0
            else:
                # Convert observation to tensor
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

                # Get action from agent
                with torch.no_grad():
                    agent = self.agents[agent_id]
                    if hasattr(agent, 'get_action'):
                        result = agent.get_action(
                            obs_tensor,
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            hidden=hiddens[agent_id],
                            deterministic=False,
                        )
                        if len(result) == 4:  # A2C returns (action, log_prob, value, hidden)
                            action, log_prob, value, hiddens[agent_id] = result
                            value = value.item()
                        else:  # REINFORCE returns (action, log_prob, hidden)
                            action, log_prob, hiddens[agent_id] = result
                            value = None

                        action = action.item()
                        log_prob = log_prob.item()
                    else:
                        # Fallback for simple agents
                        action = self.env.action_space(agent_id).sample()
                        log_prob = 0.0
                        value = 0.0
                    action_mask = 1.0

                episode_rewards[agent_id] += reward

            # Take action in environment
            self.env.step(action)
            episode_length += 1

            episode_data[agent_id]['obs'].append(obs)
            episode_data[agent_id]['actions'].append(action if action is not None else 0)
            episode_data[agent_id]['rewards'].append(reward)
            episode_data[agent_id]['dones'].append(float(done))
            episode_data[agent_id]['log_probs'].append(log_prob)
            episode_data[agent_id]['values'].append(value if value is not None else 0.0)
            episode_data[agent_id]['prev_actions'].append(prev_actions[agent_id].squeeze(0).cpu().numpy())
            episode_data[agent_id]['prev_rewards'].append(prev_rewards[agent_id].squeeze(0).cpu().numpy())
            episode_data[agent_id]['action_masks'].append(action_mask)

            if action is not None:
                one_hot = torch.zeros(1, self.env.action_space(agent_id).n, device=self.device)
                one_hot[0, action] = 1.0
                prev_actions[agent_id] = one_hot
            prev_rewards[agent_id] = torch.tensor([[reward]], dtype=torch.float32, device=self.device)

        for agent_id in self.env.possible_agents:
            self.buffer.add_episode(agent_id, episode_data[agent_id])
        self.buffer.episode_lengths.append(episode_length)

        return {
            **{f'{agent_id}_return': episode_rewards[agent_id] for agent_id in episode_rewards},
            'episode_length': episode_length,
        }

    def update_agents(self) -> Dict[str, float]:
        """Update all agents from buffer.

        Returns:
            Training metrics
        """
        all_metrics = {}

        for agent_id, agent in self.agents.items():
            # Get agent's data from buffer
            rollout_data = self.buffer.get(agent_id)

            if len(rollout_data['obs']) == 0:
                continue

            # Update agent
            metrics = agent.update(rollout_data)
            all_metrics[agent_id] = metrics

        # Clear buffer after update
        self.buffer.clear()

        return all_metrics

    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.n_episodes} episodes...")
        print(f"Agents: {list(self.agents.keys())}")
        print(f"Update frequency: {self.update_freq} episodes")

        episode_count = 0
        pbar = tqdm(range(self.n_episodes), desc="Training")

        for episode in pbar:
            # Collect episode
            episode_stats = self.collect_episode()
            self.episode_rewards.append(episode_stats)
            episode_count += 1

            # Update agents
            if episode_count % self.update_freq == 0:
                metrics = self.update_agents()
                self.training_metrics.append(metrics)

                # Update progress bar
                primary = self.env.possible_agents[0]
                avg_return = np.mean([stats[f'{primary}_return'] for stats in self.episode_rewards[-10:]])
                pbar.set_postfix({
                    'avg_return': f'{avg_return:.2f}',
                    'buffer_size': len(self.buffer),
                })

            # Evaluate
            if episode % self.eval_freq == 0 and episode > 0:
                eval_stats = self.evaluate(n_episodes=10)
                print(f"\nEpisode {episode} - Eval: {eval_stats}")

            # Save checkpoint
            if episode % self.save_freq == 0 and episode > 0:
                self.save_checkpoint(episode)

        print("\nTraining completed!")
        return self.episode_rewards, self.training_metrics

    def evaluate(self, n_episodes: int = 10) -> Dict[str, float]:
        """Evaluate agents.

        Args:
            n_episodes: Number of episodes to evaluate

        Returns:
            Evaluation statistics
        """
        eval_rewards = {agent_id: [] for agent_id in self.agents.keys()}

        for _ in range(n_episodes):
            self.env.reset()
            hiddens = {agent_id: self.agents[agent_id].info_state_encoder.init_hidden(1, self.device)
                       for agent_id in self.env.possible_agents}
            prev_actions = {agent_id: torch.zeros(1, self.env.action_space(agent_id).n, device=self.device)
                            for agent_id in self.env.possible_agents}
            prev_rewards = {agent_id: torch.zeros(1, 1, device=self.device)
                            for agent_id in self.env.possible_agents}

            episode_rewards = {agent_id: 0.0 for agent_id in self.env.possible_agents}

            for agent_id in self.env.agent_iter():
                obs, reward, termination, truncation, info = self.env.last()
                done = termination or truncation

                if done:
                    action = None
                else:
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

                    with torch.no_grad():
                        agent = self.agents[agent_id]
                        result = agent.get_action(
                            obs_tensor,
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            hidden=hiddens[agent_id],
                            deterministic=True,
                        )
                        if len(result) == 4:
                            action, _, _, hiddens[agent_id] = result
                        else:
                            action, _, hiddens[agent_id] = result
                        action = action.item()

                    episode_rewards[agent_id] += reward

                self.env.step(action)
                if action is not None:
                    one_hot = torch.zeros(1, self.env.action_space(agent_id).n, device=self.device)
                    one_hot[0, action] = 1.0
                    prev_actions[agent_id] = one_hot
                prev_rewards[agent_id] = torch.tensor([[reward]], dtype=torch.float32, device=self.device)

            for agent_id in episode_rewards:
                eval_rewards[agent_id].append(episode_rewards[agent_id])

        # Compute statistics
        stats = {}
        for agent_id in eval_rewards:
            stats[f'{agent_id}_mean'] = np.mean(eval_rewards[agent_id])
            stats[f'{agent_id}_std'] = np.std(eval_rewards[agent_id])

        return stats

    def save_checkpoint(self, episode: int):
        """Save agent checkpoints."""
        import os
        os.makedirs(self.save_path, exist_ok=True)

        for agent_id, agent in self.agents.items():
            path = f"{self.save_path}/{agent_id}_ep{episode}.pt"
            torch.save({
                'episode': episode,
                'agent_state_dict': agent.state_dict(),
                'config': self.config,
            }, path)

        print(f"\nCheckpoint saved at episode {episode}")
