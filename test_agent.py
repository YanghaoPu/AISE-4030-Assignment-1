import numpy as np
from d3qn_agent import D3QNAgent

config = {
    "gamma": 0.9,
    "learning_rate": 0.00025,
    "epsilon_start": 1.0,
    "epsilon_min": 0.1,
    "epsilon_decay": 0.999,
    "target_sync_steps": 100,
    "grad_clip": 1.0,
}

agent = D3QNAgent(
    state_dim=(4, 84, 84),
    action_dim=2,
    config=config,
    device="cpu"
)

state = np.random.rand(4, 84, 84).astype(np.float32)
next_state = np.random.rand(4, 84, 84).astype(np.float32)

action = agent.choose_action(state)
loss = agent.update(state, action, 1.0, next_state, False)

print("Action:", action)
print("Loss:", loss)
print("Epsilon:", agent.epsilon)