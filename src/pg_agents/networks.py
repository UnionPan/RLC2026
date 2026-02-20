"""Policy and value networks."""

import torch
import torch.nn as nn
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    """Policy network for discrete action spaces."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [128, 128]):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, action_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> Categorical:
        """
        Args:
            state: Information state (batch, state_dim)

        Returns:
            Categorical distribution over actions
        """
        logits = self.net(state)
        return Categorical(logits=logits)

    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        """Sample action from policy.

        Args:
            state: (batch, state_dim)
            deterministic: If True, return argmax

        Returns:
            action: (batch,)
            log_prob: (batch,)
        """
        dist = self.forward(state)
        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def evaluate_actions(self, state: torch.Tensor, actions: torch.Tensor):
        """Evaluate log prob and entropy of actions.

        Args:
            state: (batch, state_dim)
            actions: (batch,)

        Returns:
            log_probs: (batch,)
            entropy: (batch,)
        """
        dist = self.forward(state)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy


class ValueNetwork(nn.Module):
    """Value network (critic)."""

    def __init__(self, state_dim: int, hidden_dims: list = [128, 128]):
        super().__init__()
        self.state_dim = state_dim

        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (batch, state_dim)

        Returns:
            value: (batch, 1)
        """
        return self.net(state)


class ActorCriticNetwork(nn.Module):
    """Combined actor-critic network."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [128, 128],
        shared_layers: int = 1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared trunk
        shared = []
        prev_dim = state_dim
        for i, hidden_dim in enumerate(hidden_dims):
            if i >= shared_layers:
                break
            shared.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim

        self.shared = nn.Sequential(*shared) if shared else nn.Identity()

        # Policy head
        policy_layers = []
        for i, hidden_dim in enumerate(hidden_dims):
            if i < shared_layers:
                continue
            policy_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        policy_layers.append(nn.Linear(prev_dim, action_dim))
        self.policy_head = nn.Sequential(*policy_layers)

        # Value head
        value_layers = []
        prev_dim = hidden_dims[shared_layers - 1] if shared_layers > 0 else state_dim
        for i, hidden_dim in enumerate(hidden_dims):
            if i < shared_layers:
                continue
            value_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        value_layers.append(nn.Linear(prev_dim, 1))
        self.value_head = nn.Sequential(*value_layers)

    def forward(self, state: torch.Tensor):
        """
        Args:
            state: (batch, state_dim)

        Returns:
            dist: Categorical distribution over actions
            value: (batch, 1)
        """
        shared_features = self.shared(state)
        logits = self.policy_head(shared_features)
        value = self.value_head(shared_features)
        dist = Categorical(logits=logits)
        return dist, value

    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        """Sample action from policy.

        Args:
            state: (batch, state_dim)
            deterministic: If True, return argmax

        Returns:
            action: (batch,)
            log_prob: (batch,)
            value: (batch, 1)
        """
        dist, value = self.forward(state)
        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate_actions(self, state: torch.Tensor, actions: torch.Tensor):
        """Evaluate actions.

        Args:
            state: (batch, state_dim)
            actions: (batch,)

        Returns:
            log_probs: (batch,)
            values: (batch, 1)
            entropy: (batch,)
        """
        dist, value = self.forward(state)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, value, entropy


class QNetwork(nn.Module):
    """Q-network for discrete action spaces."""

    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return Q-values for all actions."""
        return self.net(state)
