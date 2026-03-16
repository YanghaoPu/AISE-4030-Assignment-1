# AISE-4030-Assignment-1
# AISE 4030 Assignment 1 – Super Mario D3QN

This project implements three Deep Q-Network based agents for **Super Mario Bros** and compares their training performance:

- **Task 1:** D3QN without Experience Replay
- **Task 2:** D3QN with Uniform Experience Replay
- **Task 3:** D3QN with Prioritized Experience Replay (PER)

All three experiments are run using the same `training_script.py` and are controlled by changing the `agent_type` in `config.yaml`.

---

## Project Structure

```text
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
