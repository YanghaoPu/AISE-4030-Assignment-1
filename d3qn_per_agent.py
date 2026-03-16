import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from per_buffer import PERBuffer
from d3qn_network import D3QNNetwork


class D3QNPERAgent:
    """
    D3QN agent with Prioritized Experience Replay (PER).
    """

    def __init__(
        self,
        state_shape: Tuple[int, ...],
        action_dim: int,
        config: Dict[str, Any],
        device: torch.device,
    ) -> None:
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.config = config
        self.device = device

        # Basic hyperparameters
        self.gamma = config["gamma"]
        self.lr = config["learning_rate"]
        self.batch_size = config["batch_size"]
        self.learning_starts = config["learning_starts"]
        self.target_sync_steps = config["target_sync_steps"]
        self.grad_clip = config.get("grad_clip", 1.0)

        # Epsilon-greedy
        self.epsilon = config["epsilon_start"]
        self.epsilon_min = config["epsilon_min"]
        self.epsilon_decay = config["epsilon_decay"]

        # PER hyperparameters
        self.per_alpha = config["per_alpha"]
        self.per_beta_start = config["per_beta_start"]
        self.per_beta = config["per_beta_start"]
        self.per_epsilon = config["per_epsilon"]
        self.beta_anneal_steps = config["beta_anneal_steps"]

        # Counters
        self.curr_step = 0
        self.learn_step = 0

        # Networks
        self.policy_net = D3QNNetwork(state_shape, action_dim).to(self.device)
        self.target_net = D3QNNetwork(state_shape, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer and elementwise loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss_fn = nn.SmoothL1Loss(reduction="none")

        # PER buffer
        self.memory = PERBuffer(
            capacity=config["replay_buffer_capacity"],
            alpha=self.per_alpha,
            epsilon=self.per_epsilon,
        )

    def act(self, state: np.ndarray) -> int:
        """
        Selects an action using epsilon-greedy exploration.
        """
        if random.random() < self.epsilon:
            action = random.randrange(self.action_dim)
        else:
            state_tensor = torch.tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                action = int(torch.argmax(q_values, dim=1).item())

        self.curr_step += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return action

    def cache(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Stores one transition in PER buffer.
        """
        self.memory.store(state, action, reward, next_state, done)

    def _anneal_beta(self) -> None:
        """
        Linearly anneals beta from beta_start to 1.0.
        """
        self.learn_step += 1
        fraction = min(1.0, self.learn_step / self.beta_anneal_steps)
        self.per_beta = self.per_beta_start + fraction * (1.0 - self.per_beta_start)

    def learn(self) -> Optional[float]:
        """
        Performs one PER-based learning step.

        Returns:
            Optional[float]: loss value if learning happens, else None.
        """
        if len(self.memory) < max(self.learning_starts, self.batch_size):
            return None

        self._anneal_beta()

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            indices,
            weights,
        ) = self.memory.sample(self.batch_size, self.per_beta)

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Current Q(s, a)
        current_q = self.policy_net(states).gather(1, actions)

        # Double DQN target:
        # policy network selects best next action
        # target network evaluates that action
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q_target = self.target_net(next_states).gather(1, next_actions)
            td_target = rewards + self.gamma * next_q_target * (1.0 - dones)

        # Weighted PER loss
        elementwise_loss = self.loss_fn(current_q, td_target)
        loss = (weights * elementwise_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
        self.optimizer.step()

        # Update priorities using fresh TD errors
        td_errors = (td_target - current_q).detach().squeeze(1).cpu().numpy()
        self.memory.update_priorities(indices, td_errors)

        # Periodically sync target network
        if self.curr_step % self.target_sync_steps == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    def save(self, path: str) -> None:
        """
        Saves policy network weights.
        """
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str) -> None:
        """
        Loads weights into both policy and target networks.
        """
        state_dict = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)