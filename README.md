# Reinforcement Learning for AI Agents & Agentic AI


**AI Engineering Internship – Task 5 (A, B, D practical component)**

A small, fully inspectable introduction to Reinforcement Learning (RL) and how it maps onto **agentic AI**. The repo trains a tabular Q-Learning agent on a classic benchmark (Taxi), then applies the same algorithm to a custom environment where an AI agent learns **which tool to call and when to stop**.

---

## Table of Contents
1. [Overview](#1-overview)
2. [Why Q-Learning?](#2-why-q-learning)
3. [Part A – Taxi (Gymnasium)](#3-part-a--taxi-gymnasium)
4. [Part B – Tool-Selection Agent](#4-part-b--tool-selection-agent)
5. [Actor / State / Action / Reward Table](#5-actor--state--action--reward-table)
6. [Limitations](#6-limitations)
7. [Project Structure](#7-project-structure)
8. [Getting Started](#8-getting-started)
9. [Next Steps](#9-next-steps)
10. [References](#10-references)

---

## 1. Overview

| Part | Goal | Environment | Algorithm |
|------|------|-------------|-----------|
| **A. Classic RL** (Task 5A/5B) | Learn the RL loop end-to-end on a standard benchmark | Gymnasium `Taxi` (v3/v4) | Tabular Q-Learning |
| **B. Agentic RL** (Task 5D) | Show an agent learning *which tool to call* and *when to stop* | Custom `ToolSelectionEnv` (Gymnasium API) | Tabular Q-Learning |

Part B is a deliberately small model of the agentic loop:

```
Agent -> observes state -> selects action -> uses tool -> receives reward -> updates policy
```

## 2. Why Q-Learning?

- Both environments have **small, discrete** state/action spaces (500×6 and 63×6), so a Q-table is exact and fully inspectable.
- Every number the agent learned can be read straight from the table, which helps in *understanding* the algorithm rather than treating it as a black box.
- It is **off-policy**: exploration (ε-greedy) and the learning target (greedy max) are cleanly separated.

**Update rule**

```
Q(s,a) <- Q(s,a) + α · [ r + γ · max_a' Q(s',a') − Q(s,a) ]
```

## 3. Part A – Taxi (Gymnasium)

| Item | Definition |
|------|-----------|
| Observation space | `Discrete(500)`: taxi position (25) × passenger location (5) × destination (4) |
| Action space | `Discrete(6)`: south, north, east, west, pickup, dropoff |
| Reward | −1 per step, +20 correct dropoff, −10 illegal pickup/dropoff |
| Termination | Passenger dropped at destination; truncation at 200 steps |
| Why suitable | Discrete, fully observable, dense-enough rewards, fast to train |

**Hyperparameters:** α = 0.1, γ = 0.99, ε: 1.0 → 0.05 (decay 0.999), 6000 episodes.

**Results (200 test episodes, greedy policy)**

| Metric | Value |
|--------|-------|
| Success rate | **100%** |
| Mean reward | **8.2** |
| Mean steps | **12.8** |
| Random-policy mean reward | −766 |

![Taxi training](results/taxi_training.png)

## 4. Part B – Tool-Selection Agent

Custom environment in [`src/tool_env.py`](src/tool_env.py).

| Item | Definition |
|------|-----------|
| State | `(query_type, stage, calls_used)`; query type ∈ {factual, math, data}; stage 0–2 = nothing / info gathered / processed |
| Actions | `web_search`, `calculator`, `code_executor`, `cache_lookup`, `ask_clarification`, `final_answer` |
| Transitions | A useful tool advances `stage` (10% random tool failure; cache hits only 30% of the time) |
| Rewards | +10 correct answer, −8 premature answer, +1 progress, −0.2 per step, per-tool cost, −0.5 wrong tool, −3 budget exhausted |
| Budget | Max 6 tool calls per episode |
| Hidden rules | factual → search; math → calculator; data → gather, then code_executor |

The agent is **never told the rules**; it must discover them from rewards alone.

**Results (500 test episodes)**

| Policy | Success rate | Mean reward | Avg tool calls |
|--------|-------------|-------------|----------------|
| Random | 16.6% | −6.1 | 3.18 |
| Always answer immediately | 0% | −8.2 | 0 |
| **Trained Q-agent** | **99.8%** | **10.06** | **1.67** |

**Learned policy (first step):**

| Query type | First action |
|-----------|--------------|
| factual | `web_search` |
| math | `calculator` |
| data | `cache_lookup` |

For data queries the agent learned to try the cheap cache first, then fall back to the proper tools.

![Tool agent training](results/tool_agent_training.png)

## 5. Actor / State / Action / Reward Table

| Actor | Current state | Possible action | Tool/Resource | Reward/Payoff | Next state |
|-------|---------------|-----------------|---------------|---------------|-----------|
| LLM Agent | math, stage 0 | call calculator | Calculator | +1 −0.2 −0.2 | math, stage 1 |
| LLM Agent | math, stage 1 | final_answer | – | +10 −0.2 | terminal (success) |
| LLM Agent | factual, stage 0 | web_search | Search API | +1 −0.2 −0.5 | factual, stage 1 |
| LLM Agent | data, stage 0 | cache_lookup | Cache | +1 −0.3 (30% hit) | data, stage 1 or stay |
| LLM Agent | data, stage 1 | code_executor | Python sandbox | +1 −1.0 | data, stage 2 |
| LLM Agent | any, stage 0 | final_answer | – | −8.2 | terminal (failure) |
| LLM Agent | any, calls = 6 | (budget out) | – | −3 | truncated |

## 6. Limitations

This toy setup is **not** a replacement for training LLM agents. Real LLM agents face:

- expensive rollouts
- sparse and delayed rewards
- huge (text) action spaces
- credit assignment over long workflows
- non-deterministic outputs and safety risks

Here the state is hand-engineered and the action set is tiny. Realistic next steps include bandit-style tool routing on top of an LLM, DQN with learned state embeddings, PPO/RLHF-style fine-tuning, and multi-agent extensions with Game Theory (Task 5E).

## 7. Project Structure

```
ai-internship-task5-rl/
├── src/
│   ├── q_learning.py        # Q-Learning agent (ε-greedy, TD update)
│   ├── train_taxi.py        # Part A: train/evaluate on Gymnasium Taxi
│   ├── tool_env.py          # Part B: custom Gymnasium environment
│   ├── train_tool_agent.py  # Part B: train/evaluate tool-selection agent
│   └── utils.py             # Plot helpers
├── results/                 # Plots, JSON metrics, saved Q-tables
├── requirements.txt
└── README.md
```

## 8. Getting Started

**Requirements:** Python 3.9+, `gymnasium>=1.0`, `numpy`, `matplotlib`

```bash
# 1. Clone the repository
git clone https://github.com/naqiather5460-ai/ai-internship-task5-rl.git
cd ai-internship-task5-rl

# 2. (Optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the experiments (from inside src/)
cd src
python train_taxi.py             # Part A
python train_tool_agent.py       # Part B
```

Outputs (plots, metrics, Q-tables) are written to `results/`.

## 9. Next Steps

- [ ] Add SARSA and DQN for comparison (Task 5A table)
- [ ] Plug in your own case-study actors/tools in `tool_env.py` (replace tools, states, rewards)
- [ ] Add a Game Theory module (payoff matrices, Nash equilibrium) for multi-agent scenarios (Task 5C/5E)
- [ ] Draw architecture, workflow, flow and state diagrams for the report

## 10. References

- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.)
- [Gymnasium documentation](https://gymnasium.farama.org)
- Watkins & Dayan (1992), *Q-learning*
