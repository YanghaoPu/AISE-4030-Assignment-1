import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from d3qn_network import D3QNNetwork


class D3QNAgent:
    """
    Double Dueling DQN agent for Task 1:
    online learning without any replay buffer.
    """

    def __init__(self, state_dim: tuple, action_dim: int, config: dict, device: str):
        """
        Initializes the D3QN agent.

        Args:
            state_dim (tuple): Observation shape as (channels, height, width).
            action_dim (int): Number of discrete actions.
            config (dict): Hyperparameter dictionary loaded from config.yaml.
            device (str): Device string, e.g. "cpu" or "cuda".

        Returns:
            None
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = torch.device(device)

        # Networks
        self.policy_net = D3QNNetwork(state_dim, action_dim).to(self.device)
        self.target_net = D3QNNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Hyperparameters
        self.gamma = config.get("gamma", 0.9)
        self.lr = config.get("learning_rate", 0.00025)

        self.epsilon = config.get("epsilon_start", 1.0)
        self.epsilon_min = config.get("epsilon_min", 0.1)
        self.epsilon_decay = config.get("epsilon_decay", 0.99999975)

        self.target_sync_steps = config.get("target_sync_steps", 10000)
        self.grad_clip = config.get("grad_clip", 1.0)

        # Optimizer and loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss_fn = nn.SmoothL1Loss()

        # Counters
        self.learn_step_counter = 0

    def choose_action(self, state: np.ndarray, evaluation_mode: bool = False) -> int:
        """
        Chooses an action using epsilon-greedy exploration.

        Args:
            state (np.ndarray): Current observation of shape (4, 84, 84).
            evaluation_mode (bool): If True, disables exploration.

        Returns:
            int: Selected action index.
        """
        if (not evaluation_mode) and (random.random() < self.epsilon):
            return random.randrange(self.action_dim)

        state_tensor = self._state_to_tensor(state)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            action = torch.argmax(q_values, dim=1).item()

        return action

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> float:
        """
        Performs one online gradient update using a single transition.

        This implements Double DQN target computation:
        1. Select next action using the policy network.
        2. Evaluate that action using the target network.
        3. Compute TD target and update policy network.

        Args:
            state (np.ndarray): Current observation.
            action (int): Action taken at current state.
            reward (float): Reward received.
            next_state (np.ndarray): Next observation.
            done (bool): Whether the episode terminated.

        Returns:
            float: Scalar loss value.
        """
        state_tensor = self._state_to_tensor(state)               # (1, C, H, W)
        next_state_tensor = self._state_to_tensor(next_state)     # (1, C, H, W)

        action_tensor = torch.tensor([[action]], dtype=torch.long, device=self.device)
        reward_tensor = torch.tensor([reward], dtype=torch.float32, device=self.device)
        done_tensor = torch.tensor([done], dtype=torch.float32, device=self.device)

        # Current Q(s, a)
        current_q = self.policy_net(state_tensor).gather(1, action_tensor).squeeze(1)

        with torch.no_grad():
            # Double DQN:
            # action selection from policy_net
            next_actions = self.policy_net(next_state_tensor).argmax(dim=1, keepdim=True)

            # action evaluation from target_net
            next_q = self.target_net(next_state_tensor).gather(1, next_actions).squeeze(1)

            target_q = reward_tensor + self.gamma * next_q * (1.0 - done_tensor)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
        self.optimizer.step()

        self.learn_step_counter += 1
        self.update_exploration_schedule()
        self._sync_target_network_if_needed()

        return loss.item()

    def update_exploration_schedule(self) -> None:
        """
        Decays epsilon after each learning step.

        Args:
            None

        Returns:
            None
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_model(self, filepath: str) -> None:
        """
        Saves the policy network weights.

        Args:
            filepath (str): Destination path.

        Returns:
            None
        """
        torch.save(
            {
                "policy_net_state_dict": self.policy_net.state_dict(),
                "target_net_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "learn_step_counter": self.learn_step_counter,
            },
            filepath,
        )

    def load_model(self, filepath: str) -> None:
        """
        Loads saved model weights and optimizer state.

        Args:
            filepath (str): Path to saved checkpoint.

        Returns:
            None
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.epsilon = checkpoint.get("epsilon", self.epsilon)
        self.learn_step_counter = checkpoint.get("learn_step_counter", 0)

    def _state_to_tensor(self, state: np.ndarray) -> torch.Tensor:
        """
        Converts an environment state into a batch tensor.

        Args:
            state (np.ndarray): Observation array of shape (4, 84, 84).

        Returns:
            torch.Tensor: Tensor of shape (1, 4, 84, 84).
        """
        if not isinstance(state, np.ndarray):
            state = np.array(state)

        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)

        if state_tensor.ndim == 3:
            state_tensor = state_tensor.unsqueeze(0)

        return state_tensor

    def _sync_target_network_if_needed(self) -> None:
        """
        Periodically copies policy network weights into the target network.

        Args:
            None

        Returns:
            None
        """
        if self.learn_step_counter % self.target_sync_steps == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())