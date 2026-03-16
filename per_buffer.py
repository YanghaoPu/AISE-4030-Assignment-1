import numpy as np
from typing import Any, List, Tuple


class SumTree:
    """
    Sum Tree data structure for Prioritized Experience Replay.

    A complete binary tree where:
    - leaf nodes store transition priorities
    - parent nodes store the sum of their children

    This allows O(log N) updates and O(log N) sampling.

    Args:
        capacity (int): Maximum number of transitions that can be stored.
    """

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")

        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data = np.empty(capacity, dtype=object)

        self.write = 0
        self.size = 0

    def total_priority(self) -> float:
        """
        Returns the total priority stored in the tree.
        This is the value at the root node.
        """
        return float(self.tree[0])

    def max_priority(self) -> float:
        """
        Returns the maximum priority currently stored among leaf nodes.
        Used for assigning initial priority to new transitions.
        """
        if self.size == 0:
            return 0.0

        leaf_start = self.capacity - 1
        leaf_end = leaf_start + self.size
        return float(np.max(self.tree[leaf_start:leaf_end]))

    def add(self, priority: float, data: Any) -> None:
        """
        Adds a new transition with the given priority.

        If the buffer is full, overwrites the oldest transition.

        Args:
            priority (float): Priority value for the new transition.
            data (Any): Transition to store.
        """
        tree_idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(tree_idx, priority)

        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, tree_idx: int, priority: float) -> None:
        """
        Updates the priority of a leaf node and propagates the change upward.

        Args:
            tree_idx (int): Index of the leaf node in the tree array.
            priority (float): New priority value.
        """
        if priority < 0:
            raise ValueError("priority must be non-negative")

        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority

        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, value: float) -> Tuple[int, float, int, Any]:
        """
        Retrieves the leaf corresponding to the given cumulative priority value.

        Args:
            value (float): A value in [0, total_priority].

        Returns:
            tuple:
                tree_idx (int): Index of the leaf in the tree.
                priority (float): Priority stored at that leaf.
                data_idx (int): Index in the data array.
                data (Any): Stored transition.
        """
        if value < 0 or value > self.total_priority():
            raise ValueError("value must be in [0, total_priority]")

        parent_idx = 0

        while True:
            left_child = 2 * parent_idx + 1
            right_child = left_child + 1

            if left_child >= len(self.tree):
                leaf_idx = parent_idx
                break

            if value <= self.tree[left_child]:
                parent_idx = left_child
            else:
                value -= self.tree[left_child]
                parent_idx = right_child

        data_idx = leaf_idx - (self.capacity - 1)
        return leaf_idx, float(self.tree[leaf_idx]), data_idx, self.data[data_idx]


class PERBuffer:
    """
    Prioritized Experience Replay buffer using a Sum Tree.

    Priority formula:
        p_i = (|td_error| + epsilon) ** alpha

    Args:
        capacity (int): Maximum buffer size.
        alpha (float): Prioritization exponent.
        epsilon (float): Small positive value to avoid zero priority.
    """

    def __init__(self, capacity: int, alpha: float = 0.6, epsilon: float = 1e-5):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")

        self.capacity = capacity
        self.alpha = alpha
        self.epsilon = epsilon
        self.tree = SumTree(capacity)

    def __len__(self) -> int:
        """Returns the number of stored transitions."""
        return self.tree.size

    def _get_priority(self, td_error: float) -> float:
        """
        Converts a TD error into a priority value.

        Args:
            td_error (float): TD error for one transition.

        Returns:
            float: priority value
        """
        return float((abs(td_error) + self.epsilon) ** self.alpha)

    def store(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool
    ) -> None:
        """
        Stores a transition in the PER buffer.

        New transitions are inserted with the current maximum priority,
        so they are sampled at least once soon after insertion.

        Args:
            state: Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state: Next state.
            done (bool): Episode termination flag.
        """
        max_p = self.tree.max_priority()
        if max_p == 0.0:
            max_p = 1.0

        transition = (state, action, reward, next_state, done)
        self.tree.add(max_p, transition)

    def sample(
        self,
        batch_size: int,
        beta: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if len(self) == 0:
            raise ValueError("Cannot sample from an empty PER buffer")

        batch = []
        indices = np.empty(batch_size, dtype=np.int32)
        priorities = np.empty(batch_size, dtype=np.float32)

        total_p = self.tree.total_priority()
        # 确保 total_p 不为 0
        if total_p == 0:
            total_p = 1.0
            
        segment = total_p / batch_size

        for i in range(batch_size):
            start = segment * i
            end = segment * (i + 1)
            value = np.random.uniform(start, end)

            tree_idx, priority, _, transition = self.tree.get_leaf(value)
            
            # 容错处理：如果拿到空数据，尝试重新随机取一个
            while transition is None:
                value = np.random.uniform(0, total_p)
                tree_idx, priority, _, transition = self.tree.get_leaf(value)

            indices[i] = tree_idx
            priorities[i] = priority
            batch.append(transition)

        # 重要性采样权重计算
        # 增加 epsilon 防止 sampling_probs 为 0 导致除零 [cite: 9]
        sampling_probs = priorities / total_p
        # 加上一个小值 1e-8 确保稳定性
        weights = (len(self) * (sampling_probs + 1e-8)) ** (-beta)
        
        # 归一化权重
        max_weight = weights.max()
        if max_weight == 0:
            max_weight = 1.0
        weights /= max_weight

        # 解包 batch
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
            indices,
            weights.astype(np.float32),
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """
        Updates priorities of sampled transitions after a learning step.

        Args:
            indices (np.ndarray): Tree indices returned by sample().
            td_errors (np.ndarray): New TD errors for those sampled transitions.
        """
        if len(indices) != len(td_errors):
            raise ValueError("indices and td_errors must have the same length")

        for idx, td_error in zip(indices, td_errors):
            priority = self._get_priority(float(td_error))
            self.tree.update(int(idx), priority)