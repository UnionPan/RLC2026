"""Rollout buffer for collecting trajectories."""

from typing import Dict, List, Optional, Any
import torch
import numpy as np


class RolloutBuffer:
    """Buffer for storing episode trajectories."""

    def __init__(
        self,
        buffer_size: int,
        obs_shape: tuple,
        action_dim: int,
        device: str = 'cpu',
    ):
        self.buffer_size = buffer_size
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.device = device

        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

        self.ptr = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        log_prob: Optional[float] = None,
        value: Optional[float] = None,
    ):
        """Add a single transition."""
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

        if log_prob is not None:
            self.log_probs.append(log_prob)
        if value is not None:
            self.values.append(value)

        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.full = True

    def get(self) -> Dict[str, torch.Tensor]:
        """Get all data and clear buffer.

        Returns:
            Dictionary with keys: obs, actions, rewards, dones, log_probs, values
        """
        data = {
            'obs': torch.as_tensor(np.array(self.observations), dtype=torch.float32, device=self.device),
            'actions': torch.as_tensor(np.array(self.actions), dtype=torch.long, device=self.device),
            'rewards': torch.as_tensor(np.array(self.rewards), dtype=torch.float32, device=self.device),
            'dones': torch.as_tensor(np.array(self.dones), dtype=torch.float32, device=self.device),
        }

        if self.log_probs:
            data['log_probs'] = torch.as_tensor(np.array(self.log_probs), dtype=torch.float32, device=self.device)

        if self.values:
            data['values'] = torch.as_tensor(np.array(self.values), dtype=torch.float32, device=self.device)

        return data

    def clear(self):
        """Clear the buffer."""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.ptr = 0
        self.full = False

    def __len__(self):
        return len(self.observations)


class MultiAgentRolloutBuffer:
    """Buffer for multi-agent environments with variable-length episodes."""

    def __init__(
        self,
        buffer_size: int,
        obs_shape: tuple,
        action_dim: int,
        agent_ids: List[str],
        device: str = 'cpu',
    ):
        self.buffer_size = buffer_size
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.device = device

        self.agent_ids = list(agent_ids)
        self.episodes = {agent_id: [] for agent_id in self.agent_ids}
        self.episode_rewards = {agent_id: [] for agent_id in self.agent_ids}
        self.episode_lengths = []

    def add_episode(self, agent_id: str, episode_data: Dict[str, List[Any]]):
        """Add a completed episode for a specific agent."""
        self.episodes[agent_id].append(episode_data)
        self.episode_rewards[agent_id].append(sum(episode_data['rewards']))

    def _pad_sequence(self, seq: List[Any], max_len: int, fill_value: Any):
        padded = list(seq) + [fill_value] * (max_len - len(seq))
        return padded

    def get(self, agent_id: str) -> Dict[str, torch.Tensor]:
        """Get padded data for a specific agent."""
        episodes = self.episodes.get(agent_id, [])
        if not episodes:
            return {'obs': torch.empty(0, device=self.device)}

        max_len = max(len(ep['actions']) for ep in episodes)
        n_episodes = len(episodes)

        obs_batch = np.zeros((n_episodes, max_len, *self.obs_shape), dtype=np.float32)
        actions_batch = np.zeros((n_episodes, max_len), dtype=np.int64)
        rewards_batch = np.zeros((n_episodes, max_len), dtype=np.float32)
        dones_batch = np.ones((n_episodes, max_len), dtype=np.float32)
        log_probs_batch = np.zeros((n_episodes, max_len), dtype=np.float32)
        values_batch = np.zeros((n_episodes, max_len), dtype=np.float32)
        prev_actions_batch = np.zeros((n_episodes, max_len, self.action_dim), dtype=np.float32)
        prev_rewards_batch = np.zeros((n_episodes, max_len, 1), dtype=np.float32)
        action_masks_batch = np.zeros((n_episodes, max_len), dtype=np.float32)

        for i, ep in enumerate(episodes):
            length = len(ep['actions'])
            obs_batch[i, :length] = np.array(ep['obs'], dtype=np.float32)
            actions_batch[i, :length] = np.array(ep['actions'], dtype=np.int64)
            rewards_batch[i, :length] = np.array(ep['rewards'], dtype=np.float32)
            dones_batch[i, :length] = np.array(ep['dones'], dtype=np.float32)
            log_probs_batch[i, :length] = np.array(ep['log_probs'], dtype=np.float32)
            values_batch[i, :length] = np.array(ep['values'], dtype=np.float32)
            prev_actions_batch[i, :length] = np.array(ep['prev_actions'], dtype=np.float32)
            prev_rewards_batch[i, :length] = np.array(ep['prev_rewards'], dtype=np.float32)
            action_masks_batch[i, :length] = np.array(ep['action_masks'], dtype=np.float32)

        data = {
            'obs': torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device),
            'actions': torch.as_tensor(actions_batch, dtype=torch.long, device=self.device),
            'rewards': torch.as_tensor(rewards_batch, dtype=torch.float32, device=self.device),
            'dones': torch.as_tensor(dones_batch, dtype=torch.float32, device=self.device),
            'log_probs': torch.as_tensor(log_probs_batch, dtype=torch.float32, device=self.device),
            'values': torch.as_tensor(values_batch, dtype=torch.float32, device=self.device),
            'prev_actions': torch.as_tensor(prev_actions_batch, dtype=torch.float32, device=self.device),
            'prev_rewards': torch.as_tensor(prev_rewards_batch, dtype=torch.float32, device=self.device),
            'action_masks': torch.as_tensor(action_masks_batch, dtype=torch.float32, device=self.device),
        }

        return data

    def get_all(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return {agent_id: self.get(agent_id) for agent_id in self.agent_ids}

    def clear(self):
        """Clear all stored episodes."""
        self.episodes = {agent_id: [] for agent_id in self.agent_ids}
        self.episode_rewards = {agent_id: [] for agent_id in self.agent_ids}
        self.episode_lengths = []

    def get_episode_stats(self) -> Dict[str, float]:
        stats = {}
        for agent_id, rewards in self.episode_rewards.items():
            if rewards:
                stats[f'{agent_id}_return'] = np.mean(rewards)

        if self.episode_lengths:
            stats['episode_length'] = np.mean(self.episode_lengths)

        return stats

    def __len__(self):
        total = 0
        for episodes in self.episodes.values():
            total += sum(len(ep['actions']) for ep in episodes)
        return total


class ReplayBuffer:
    """Experience replay buffer for off-policy algorithms (QMIX, DQN)."""

    def __init__(
        self,
        capacity: int,
        num_agents: int,
        obs_dim: int,
        global_state_dim: int,
        device: str = 'cpu',
    ):
        """
        Args:
            capacity: Maximum buffer size
            num_agents: Number of agents
            obs_dim: Observation dimension per agent
            global_state_dim: Global state dimension
            device: torch device
        """
        self.capacity = capacity
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.global_state_dim = global_state_dim
        self.device = device

        # Pre-allocate memory for efficiency
        self.observations = {
            i: torch.zeros((capacity, obs_dim), dtype=torch.float32)
            for i in range(num_agents)
        }
        self.next_observations = {
            i: torch.zeros((capacity, obs_dim), dtype=torch.float32)
            for i in range(num_agents)
        }
        self.actions = {
            i: torch.zeros((capacity,), dtype=torch.long)
            for i in range(num_agents)
        }
        self.rewards = torch.zeros((capacity,), dtype=torch.float32)
        self.dones = torch.zeros((capacity,), dtype=torch.float32)
        self.global_states = torch.zeros((capacity, global_state_dim), dtype=torch.float32)
        self.next_global_states = torch.zeros((capacity, global_state_dim), dtype=torch.float32)

        self.position = 0
        self.size = 0

    def add(
        self,
        observations: Dict[int, torch.Tensor],
        actions: Dict[int, int],
        reward: float,
        next_observations: Dict[int, torch.Tensor],
        done: bool,
        global_state: torch.Tensor,
        next_global_state: torch.Tensor,
    ):
        """Add a multi-agent transition.

        Args:
            observations: Dict {agent_id: observation}
            actions: Dict {agent_id: action}
            reward: Team reward (shared)
            next_observations: Dict {agent_id: next_observation}
            done: Episode done flag
            global_state: Global state
            next_global_state: Next global state
        """
        idx = self.position

        for i in range(self.num_agents):
            self.observations[i][idx] = observations[i].cpu()
            self.next_observations[i][idx] = next_observations[i].cpu()
            self.actions[i][idx] = actions[i]

        self.rewards[idx] = reward
        self.dones[idx] = float(done)
        self.global_states[idx] = global_state.cpu()
        self.next_global_states[idx] = next_global_state.cpu()

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, Any]:
        """Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Dict with batched transitions
        """
        indices = np.random.choice(self.size, batch_size, replace=False)

        batch = {
            'observations': {
                i: self.observations[i][indices].to(self.device)
                for i in range(self.num_agents)
            },
            'actions': {
                i: self.actions[i][indices].to(self.device)
                for i in range(self.num_agents)
            },
            'rewards': self.rewards[indices].to(self.device),
            'next_observations': {
                i: self.next_observations[i][indices].to(self.device)
                for i in range(self.num_agents)
            },
            'dones': self.dones[indices].to(self.device),
            'global_states': self.global_states[indices].to(self.device),
            'next_global_states': self.next_global_states[indices].to(self.device),
        }

        return batch

    def __len__(self):
        """Return current buffer size."""
        return self.size
