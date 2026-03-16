import torch
import torch.nn as nn


class D3QNNetwork(nn.Module):
    """
    Double Dueling Deep Q-Network for Super Mario Bros.

    The network takes a stacked grayscale observation of shape
    (4, 84, 84) and outputs one Q-value per discrete action.
    """

    def __init__(self, input_dim: tuple, action_dim: int):
        """
        Initializes the D3QN network.

        Args:
            input_dim (tuple): Observation shape as (channels, height, width).
                Expected shape for this assignment is (4, 84, 84).
            action_dim (int): Number of discrete actions.

        Returns:
            None
        """
        super().__init__()

        c, h, w = input_dim

        if (h, w) != (84, 84):
            raise ValueError(
                f"Expected input height/width of (84, 84), but got {(h, w)}"
            )

        # Shared convolutional feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),

            nn.Flatten()
        )

        # Dynamically determine flattened feature size
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            feature_dim = self.feature_extractor(dummy).shape[1]

        # Value stream V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        # Advantage stream A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass and returns Q-values for each action.

        Args:
            x (torch.Tensor): Input tensor of shape
                (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Q-values of shape (batch_size, action_dim).
        """
        x = x.float()

        features = self.feature_extractor(x)

        value = self.value_stream(features)              # (B, 1)
        advantage = self.advantage_stream(features)      # (B, A)

        # Dueling aggregation
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_values