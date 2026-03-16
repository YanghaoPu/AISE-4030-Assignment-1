import torch
from d3qn_network import D3QNNetwork

net = D3QNNetwork(input_dim=(4, 84, 84), action_dim=2)

x = torch.randn(8, 4, 84, 84)
y = net(x)

print("Output shape:", y.shape)
print(y)