# Temporal Difference Learning: SARSA vs Q-Learning

This assignment implements and compares two fundamental temporal difference learning algorithms - SARSA and Q-learning - on a 5×5 grid world environment.

## 📋 Overview

The project demonstrates the key differences between **on-policy** (SARSA) and **off-policy** (Q-learning) temporal difference learning methods in a reinforcement learning context.

### Environment Description

![Grid World](grid_world.png)

- **5×5 Grid World** with different types of states:
  - 🟦 **Start State** (Blue): Agent begins here at position (4,0)
  - 🟥 **Penalty States** (Red): Give -20 reward and reset agent to start
  - ⬜ **Regular States** (White): Give -1 reward for each move
  - ⬛ **Terminal States** (Black): End the episode (goal states)

### Algorithms Implemented

1. **SARSA (State-Action-Reward-State-Action)**

   - On-policy algorithm
   - Updates Q-values using the action actually taken
   - More conservative, learns about the policy being followed

2. **Q-Learning**
   - Off-policy algorithm
   - Updates Q-values using the maximum Q-value for next state
   - More aggressive, learns about the optimal policy

## 🚀 Quick Start

### Prerequisites

```bash
pip install jax numpy matplotlib jupyter
```

### Running the Experiment

1. **Clone or download** the repository
2. **Navigate** to the `temporal_difference_learning` directory
3. **Open** the Jupyter notebook:

   ```bash
   jupyter notebook notebook_a3.ipynb
   ```

4. **Run all cells** to reproduce the results

### Key Differences Observed

- **SARSA** takes a more conservative path, avoiding risky areas
- **Q-Learning** takes a more direct but riskier path through penalty zones
- Both algorithms achieve similar final performance but use different strategies

## 🔧 Customization

### Hyperparameters

You can modify these parameters in the training functions:

```python
# Training parameters
n_episodes = 1000    # Number of training episodes
alpha = 0.1          # Learning rate
gamma = 0.95         # Discount factor
epsilon = 0.1        # Exploration rate (ε-greedy)
seed = 42            # Random seed for reproducibility
```

### Environment Modifications

To change the grid layout, modify the `create_grid_layout()` function:

```python
def create_grid_layout():
    grid = jnp.array([
        [1, 0, 0, 0, 1],  # 1=terminal, 0=empty
        [0, 0, 0, 0, 0],  # 2=penalty, 3=start
        [2, 2, 0, 2, 2],
        [0, 0, 0, 0, 0],
        [3, 0, 0, 0, 0],
    ])
    return grid
```

## 📈 Visualizations

The notebook generates several plots:

1. **Learning Curves**: Episode rewards over time for both algorithms
2. **Moving Average**: Smoothed performance comparison
3. **Policy Trajectories**: Visual representation of learned paths on the grid

## 🧮 Mathematical Foundations

### SARSA Update Rule

$$Q(s,a) ← Q(s,a) + \alpha[r + \gamma Q(s',a') - Q(s,a)]$$

### Q-Learning Update Rule

$$Q(s,a) ← Q(s,a) + \alpha[r + \gamma  max Q(s',a') - Q(s,a)]$$

Where:

- `s`, `a`: current state and action
- `r`: immediate reward
- `γ`: discount factor
- `α`: learning rate
- `s'`, `a'`: next state and next action

## 🔍 Implementation Details

### Key Features

- **JAX Implementation**: Uses JAX for efficient numerical computation and JIT compilation
- **Epsilon-Greedy Exploration**: Balances exploration vs exploitation
- **Epsilon Decay**: Gradually reduces exploration over time
- **Modular Design**: Clean separation between environment, algorithms, and analysis

### Code Structure

```text
notebook_a3.ipynb
├── Environment Setup (Grid world, state transitions, rewards)
├── Q-Table Operations (Creation, updates, policy extraction)
├── SARSA Implementation (Training loop, episode runner)
├── Q-Learning Implementation (Training loop, episode runner)
├── Analysis & Comparison (Trajectory extraction, performance metrics)
└── Visualization (Learning curves, policy paths)
```

## 📝 Author

**Isaac Edem Adoboe**

- Email: [ieadoboe@mun.ca](mailto:ieadoboe@mun.ca)

## 🤝 Contributing

Feel free to experiment with:

- Different grid layouts
- Alternative exploration strategies
- Additional hyperparameter combinations
- Extended analysis and visualizations

## 📚 References

- Sutton, R. S., & Barto, A. G. (2018). _Reinforcement Learning: An Introduction_
- JAX Documentation: [https://jax.readthedocs.io/](https://jax.readthedocs.io/)
