from training_script import make_mario_env
import numpy as np

env = make_mario_env("SuperMarioBros-1-1-v3", render_mode=None, seed=42)

obs, info = env.reset()
obs = np.array(obs)

print("Environment created successfully!")
print("Observation shape:", obs.shape)
print("Action space:", env.action_space)

env.close()

print("Setup is complete!")