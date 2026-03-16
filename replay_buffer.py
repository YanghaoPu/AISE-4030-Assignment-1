import random
import numpy as np
from typing import Tuple

class ReplayBuffer:
    """
    Uniform Experience Replay Buffer for storing and sampling transitions.
    """

    def __init__(self, capacity: int):
        """
        Initializes the uniform replay buffer.

        Args:
            capacity (int): Maximum number of transitions to store.

        Returns:
            None
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Stores a new transition in the buffer. Overwrites the oldest transition if full.

        Args:
            state (np.ndarray): Current state observation.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (np.ndarray): Next state observation.
            done (bool): Whether the episode terminated.

        Returns:
            None
        """
        # If buffer is not full yet, expand it
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
            
        # Overwrite the oldest transition (circular buffer)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Samples a random mini-batch of transitions uniformly from the buffer.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            tuple: (states, actions, rewards, next_states, dones) as numpy arrays.
        """
        # Sample uniformly at random
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self) -> int:
        """
        Returns the current number of stored transitions.

        Args:
            None

        Returns:
            int: Current buffer size.
        """
        return len(self.buffer)