import os
import yaml
import gym
import gym_super_mario_bros
import numpy as np
import torch
from nes_py.wrappers import JoypadSpace
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack

from d3qn_agent import D3QNAgent
from d3qn_er_agent import D3QNERAgent
from d3qn_per_agent import D3QNPERAgent


class SqueezeObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        old_shape = self.observation_space.shape
        if len(old_shape) == 3 and old_shape[-1] == 1:
            self.observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=old_shape[:2],
                dtype=self.observation_space.dtype
            )

    def observation(self, obs):
        if hasattr(obs, "shape") and len(obs.shape) == 3 and obs.shape[-1] == 1:
            return obs.squeeze(-1)
        return obs


def make_mario_env(env_name, render_mode=None, seed=42):
    env = gym_super_mario_bros.make(
        env_name,
        apply_api_compatibility=True,
        disable_env_checker=True
    )

    simple_movement = [["right"], ["right", "A"]]
    env = JoypadSpace(env, simple_movement)
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, (84, 84))
    env = SqueezeObsWrapper(env)
    env = FrameStack(env, 4)

    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    return env


def build_agent(agent_type, state_shape, action_dim, config, device):
    if agent_type == "d3qn":
        return D3QNAgent(state_shape, action_dim, config, device)
    elif agent_type == "d3qn_er":
        return D3QNERAgent(state_shape, action_dim, config, device)
    elif agent_type == "d3qn_per":
        return D3QNPERAgent(state_shape, action_dim, config, device)
    else:
        raise ValueError(f"Unknown agent_type: {agent_type}")


def get_save_dir(agent_type):
    if agent_type == "d3qn":
        return "d3qn_results"
    elif agent_type == "d3qn_er":
        return "d3qn_er_results"
    elif agent_type == "d3qn_per":
        return "d3qn_per_results"
    else:
        raise ValueError(f"Unknown agent_type: {agent_type}")


def save_agent(agent, path):
    if hasattr(agent, "save_model"):
        agent.save_model(path)
    elif hasattr(agent, "save"):
        agent.save(path)
    else:
        print("Warning: agent has no save function.")


def get_action(agent, state, agent_type):
    if agent_type == "d3qn_per":
        return agent.act(state)
    else:
        return agent.choose_action(state, evaluation_mode=False)


def train_step(agent, agent_type, state, action, reward, next_state, done):
    if agent_type == "d3qn_per":
        agent.cache(state, int(action), float(reward), next_state, done)
        return agent.learn()
    else:
        return agent.update(state, action, reward, next_state, done)


def get_progress_info(agent, agent_type):
    if agent_type == "d3qn_per":
        beta_val = getattr(agent, "per_beta", None)
        return f"Beta: {beta_val:.3f}" if beta_val is not None else ""
    else:
        epsilon_val = getattr(agent, "epsilon", None)
        return f"Epsilon: {epsilon_val:.4f}" if epsilon_val is not None else ""


def train():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    agent_type = config.get("agent_type", "d3qn")
    save_dir = get_save_dir(agent_type)
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = config.get("seed", 42)

    env = make_mario_env(
        config["env_name"],
        render_mode=None,
        seed=seed
    )

    state, info = env.reset()
    state = np.array(state)

    print("Environment created successfully!")
    print("Observation shape:", state.shape)
    print("Action space:", env.action_space)

    agent = build_agent(agent_type, state.shape, env.action_space.n, config, device)

    print(f"Starting training for {agent_type} on {device}...")
    print(f"Results will be saved to: {save_dir}/")

    all_rewards = []
    all_losses = []
    all_steps = []

    num_episodes = config["num_episodes"]
    max_steps = config.get("max_steps_per_episode", 5000)
    save_every = config.get("save_every", 500)

    for episode in range(1, num_episodes + 1):
        state, info = env.reset()
        state = np.array(state)

        done = False
        episode_reward = 0.0
        episode_losses = []
        step_count = 0

        while not done and step_count < max_steps:
            action = get_action(agent, state, agent_type)

            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = np.array(next_state)
            done = terminated or truncated

            loss = train_step(agent, agent_type, state, action, reward, next_state, done)

            if loss is not None:
                episode_losses.append(loss)

            episode_reward += reward
            state = next_state
            step_count += 1

        avg_loss = float(np.mean(episode_losses)) if episode_losses else 0.0
        all_rewards.append(episode_reward)
        all_losses.append(avg_loss)
        all_steps.append(step_count)

        progress_info = get_progress_info(agent, agent_type)
        print(
            f"Episode {episode}/{num_episodes} | "
            f"Steps: {step_count} | "
            f"Reward: {episode_reward:.2f} | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"{progress_info}"
        )

        if episode % save_every == 0 or episode == num_episodes:
            save_agent(agent, os.path.join(save_dir, f"model_ep_{episode}.pt"))
            np.save(os.path.join(save_dir, "rewards_history.npy"), np.array(all_rewards))
            np.save(os.path.join(save_dir, "losses_history.npy"), np.array(all_losses))
            np.save(os.path.join(save_dir, "steps_history.npy"), np.array(all_steps))

    env.close()
    print("Training complete.")


if __name__ == "__main__":
    train()