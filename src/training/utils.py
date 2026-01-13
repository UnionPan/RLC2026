"""Training utilities for multi-agent RL."""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


def get_device(device: str = 'cuda', verbose: bool = True) -> str:
    """Get torch device with automatic CPU fallback.

    Args:
        device: Requested device ('cuda' or 'cpu')
        verbose: Print device info

    Returns:
        Device string ('cuda' or 'cpu')
    """
    if device == 'cuda':
        if torch.cuda.is_available():
            if verbose:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"✓ Using GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            return 'cuda'
        else:
            if verbose:
                print("⚠ CUDA requested but not available, using CPU")
            return 'cpu'
    else:
        if verbose:
            print(f"• Using device: {device}")
        return device


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation (GAE).

    Args:
        rewards: Reward sequence (T,)
        values: Value predictions (T,)
        dones: Done flags (T,)
        next_value: Bootstrap value for last state (scalar)
        gamma: Discount factor
        gae_lambda: GAE lambda parameter

    Returns:
        advantages: GAE advantages (T,)
        returns: Discounted returns (T,)
    """
    T = len(rewards)
    advantages = torch.zeros_like(rewards)
    gae = 0

    # Backward pass to compute advantages
    for t in reversed(range(T)):
        if t == T - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]

        # TD error: delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]

        # GAE: A_t = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages[t] = gae

    # Returns are advantages + values
    returns = advantages + values

    return advantages, returns


def normalize_advantages(advantages: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize advantages to have mean 0 and std 1.

    Args:
        advantages: Advantage tensor
        eps: Small epsilon for numerical stability

    Returns:
        Normalized advantages
    """
    return (advantages - advantages.mean()) / (advantages.std() + eps)


class RunningMeanStd:
    """Running mean and standard deviation calculator.

    Uses Welford's online algorithm for numerical stability.
    """

    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        """
        Args:
            epsilon: Small value to avoid division by zero
            shape: Shape of the data
        """
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray):
        """Update statistics with new data.

        Args:
            x: New data point(s)
        """
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(
        self,
        batch_mean: np.ndarray,
        batch_var: np.ndarray,
        batch_count: int,
    ):
        """Update from batch statistics.

        Args:
            batch_mean: Mean of batch
            batch_var: Variance of batch
            batch_count: Number of samples in batch
        """
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
        """Normalize data using running statistics.

        Args:
            x: Data to normalize
            epsilon: Small value for numerical stability

        Returns:
            Normalized data
        """
        return (x - self.mean) / np.sqrt(self.var + epsilon)


class RewardNormalizer:
    """Normalize rewards using running statistics."""

    def __init__(self, gamma: float = 0.99, epsilon: float = 1e-8):
        """
        Args:
            gamma: Discount factor for return calculation
            epsilon: Small value for numerical stability
        """
        self.gamma = gamma
        self.epsilon = epsilon
        self.running_stats = RunningMeanStd(shape=())
        self.returns = 0.0

    def normalize(self, reward: float, done: bool = False) -> float:
        """Normalize a single reward.

        Args:
            reward: Raw reward
            done: Whether episode is done

        Returns:
            Normalized reward
        """
        self.returns = self.returns * self.gamma + reward
        self.running_stats.update(np.array([self.returns]))

        if done:
            self.returns = 0.0

        return reward / np.sqrt(self.running_stats.var + self.epsilon)

    def normalize_batch(self, rewards: np.ndarray) -> np.ndarray:
        """Normalize a batch of rewards.

        Args:
            rewards: Array of rewards

        Returns:
            Normalized rewards
        """
        self.running_stats.update(rewards)
        return rewards / np.sqrt(self.running_stats.var + self.epsilon)


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Calculate explained variance.

    Measures how well predictions explain the variance in true values.
    1.0 = perfect prediction, 0.0 = as good as predicting the mean,
    negative = worse than predicting the mean.

    Args:
        y_pred: Predicted values
        y_true: True values

    Returns:
        Explained variance
    """
    var_y = np.var(y_true)
    if var_y == 0:
        return np.nan
    return 1 - np.var(y_true - y_pred) / var_y


def polyak_update(
    source: nn.Module,
    target: nn.Module,
    tau: float = 0.005,
):
    """Polyak averaging update for target networks.

    target = (1 - tau) * target + tau * source

    Args:
        source: Source network (online network)
        target: Target network
        tau: Interpolation parameter (0 = no update, 1 = full copy)
    """
    with torch.no_grad():
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.mul_(1 - tau)
            target_param.data.add_(tau * source_param.data)


def hard_update(source: nn.Module, target: nn.Module):
    """Hard update: copy source weights to target.

    Args:
        source: Source network
        target: Target network
    """
    target.load_state_dict(source.state_dict())


class LinearSchedule:
    """Linear interpolation schedule."""

    def __init__(
        self,
        start_value: float,
        end_value: float,
        duration: int,
    ):
        """
        Args:
            start_value: Initial value
            end_value: Final value
            duration: Number of steps to interpolate over
        """
        self.start_value = start_value
        self.end_value = end_value
        self.duration = duration

    def __call__(self, step: int) -> float:
        """Get value at given step.

        Args:
            step: Current step

        Returns:
            Interpolated value
        """
        if step >= self.duration:
            return self.end_value

        fraction = step / self.duration
        return self.start_value + fraction * (self.end_value - self.start_value)


def get_linear_fn(start: float, end: float, end_fraction: float = 1.0):
    """Create a linear schedule function.

    Args:
        start: Start value
        end: End value
        end_fraction: When to reach end value (0-1)

    Returns:
        Function that takes progress (0-1) and returns scheduled value
    """
    def func(progress: float) -> float:
        if progress >= end_fraction:
            return end
        else:
            return start + (end - start) * progress / end_fraction

    return func


def discount_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    """Compute discounted cumulative sum.

    output[i] = x[i] + gamma * x[i+1] + gamma^2 * x[i+2] + ...

    Args:
        x: Input array
        gamma: Discount factor

    Returns:
        Discounted cumulative sum
    """
    result = np.zeros_like(x)
    result[-1] = x[-1]

    for t in reversed(range(len(x) - 1)):
        result[t] = x[t] + gamma * result[t + 1]

    return result


def set_seed(seed: int):
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)

    # Make PyTorch deterministic (slower)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EpsilonScheduler:
    """Epsilon-greedy exploration schedule."""

    def __init__(
        self,
        start: float = 1.0,
        end: float = 0.05,
        decay_steps: int = 50000,
    ):
        """
        Args:
            start: Initial epsilon
            end: Final epsilon
            decay_steps: Number of steps to decay over
        """
        self.start = start
        self.end = end
        self.decay_steps = decay_steps
        self.current_step = 0

    def get_epsilon(self) -> float:
        """Get current epsilon value."""
        if self.current_step >= self.decay_steps:
            return self.end

        # Linear decay
        epsilon = self.start - (self.start - self.end) * self.current_step / self.decay_steps
        return epsilon

    def step(self):
        """Increment step counter."""
        self.current_step += 1

    def reset(self):
        """Reset to initial epsilon."""
        self.current_step = 0
