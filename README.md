AISE 4030 Assignment 1 - Super Mario D3QN

_README document export_

This project implements three Deep Q-Network based agents for Super Mario Bros and compares their training performance.

# Tasks

- Task 1: D3QN without Experience Replay
- Task 2: D3QN with Uniform Experience Replay
- Task 3: D3QN with Prioritized Experience Replay (PER)

All three experiments are run using the same training_script.py and are controlled by changing the agent_type in config.yaml.

# Project Structure

AISE-4030-Assignment-1/  
│── config.yaml  
│── environment.py  
│── d3qn_network.py  
│── d3qn_agent.py  
│── d3qn_er_agent.py  
│── d3qn_per_agent.py  
│── replay_buffer.py  
│── per_buffer.py  
│── training_script.py  
│── utils.py  
│── README.md  
│── requirements.txt  
│  
├── d3qn_results/  
├── d3qn_er_results/  
└── d3qn_per_results/

# Environment

The environment used in this project is SuperMarioBros-1-1-v3.

# Preprocessing

The observation preprocessing pipeline includes:

- JoypadSpace with simplified action space
- grayscale observation
- resize to 84 x 84
- frame stack of 4

# Observation Space

Final observation shape: (4, 84, 84)

# Action Space

Final action space: Discrete(2)

# Installation

Create the environment and install dependencies:

conda create -n AISE4030 python=3.10 -y  
conda activate AISE4030  
pip install -r requirements.txt

# How to Run

All experiments use the same script:

python training_script.py

# Config Setting

To switch between tasks, change agent_type in config.yaml.

## Task 1

agent_type: d3qn

## Task 2

agent_type: d3qn_er

## Task 3

agent_type: d3qn_per

# Output Folders

- d3qn_results/ → Task 1 outputs
- d3qn_er_results/ → Task 2 outputs
- d3qn_per_results/ → Task 3 outputs

# Example Output Files

Typical output files include:

episode_rewards.npy  
episode_losses.npy  
reward_curve.png  
loss_curve.png  
model_ep_500.pt

Some large .pt model files are not uploaded to GitHub because of file size limits.

# Notes

- Training may take a long time depending on hardware.
- GPU is recommended for faster training.
- The final report compares reward and loss curves across the three agents.
- All agents share the same network architecture and environment setup.

# Team Contribution

- Task 1: d3qn_agent.py
- Task 2: replay_buffer.py, d3qn_er_agent.py
- Task 3: per_buffer.py, d3qn_per_agent.py

# Shared Files

- environment.py
- d3qn_network.py
- training_script.py
- config.yaml
- utils.py

# Course Information

**Course:** AISE 4030 - Reinforcement Learning

**Assignment:** Assignment 1 - Deep Q-Networks
