import gym
import simple_driving
import numpy as np
import random
import pickle
import os
from collections import defaultdict
import matplotlib.pyplot as plt

# -----------------------
# Configurable Parameters
# -----------------------
EPISODES = 5000
MAX_STEPS = 200
BINS = 20
ALPHA = 0.1
GAMMA = 0.99
EPSILON_DECAY = 0.999
EPSILON_MIN = 0.05
EPSILON_START = 1.0
SAVE_EVERY = 1000
QTABLE_FILE = "qtable.pkl"
EPSILON_FILE = "epsilon.pkl"

# -----------------------
# Discretize State
# -----------------------
"""
def discretize(state, bins=15, low=-10, high=10):
    xg, yg, xo, yo = state
    dxg = int(np.clip((xg - low) / (high - low) * bins, 0, bins - 1))
    dyg = int(np.clip((yg - low) / (high - low) * bins, 0, bins - 1))
    dxo = int(np.clip((xo - low) / (high - low) * bins, 0, bins - 1))
    dyo = int(np.clip((yo - low) / (high - low) * bins, 0, bins - 1))
    return (dxg, dyg, dxo, dyo)
"""

def discretize(state, bins=15, low=-10, high=10):
    xg, yg = state[:2]  # Assume input is (xg, yg, ...) but ignore anything after
    dxg = int(np.clip((xg - low) / (high - low) * bins, 0, bins - 1))
    dyg = int(np.clip((yg - low) / (high - low) * bins, 0, bins - 1))
    return (dxg, dyg)

# -----------------------
# Biased Exploration
# -----------------------
def biased_exploration():
    raw_probs = np.array([
        0.04,   # Reverse-Left
        0.04,   # Reverse
        0.04,   # Reverse-Right
        0.03,   # Steer-Left (limited)
        0.005,  # Stationary (discouraged)
        0.03,   # Steer-Right (limited)
        0.26,   # Forward-Right
        0.38,   # Forward
        0.20    # Forward-Left
    ])
    probs = raw_probs / raw_probs.sum()  # Normalize to ensure sum = 1
    return np.random.choice(np.arange(9), p=probs)


# -----------------------
# Load Previous Progress
# -----------------------
if os.path.exists(QTABLE_FILE):
    with open(QTABLE_FILE, "rb") as f:
        q_table = defaultdict(lambda: np.zeros(9), pickle.load(f))
    print("✅ Loaded previous Q-table.")
else:
    q_table = defaultdict(lambda: np.zeros(9))
    print("🔁 Starting with new Q-table.")

if os.path.exists(EPSILON_FILE):
    with open(EPSILON_FILE, "rb") as f:
        epsilon = pickle.load(f)
    print(f"✅ Resuming with previous epsilon: {epsilon:.3f}")
else:
    epsilon = EPSILON_START
    print("🔁 Starting with new epsilon.")

# -----------------------
# Environment Setup
# -----------------------
env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=False, isDiscrete=True)
episode_rewards = []

# -----------------------
# Training Loop
# -----------------------
for ep in range(EPISODES):
    state, _ = env.reset()
    state = discretize(state)

    total_reward = 0

    for t in range(MAX_STEPS):
        if random.random() < epsilon:
            action = biased_exploration()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, done, _, _ = env.step(action)
        next_state = discretize(next_state)

        q_table[state][action] += ALPHA * (
            reward + GAMMA * np.max(q_table[next_state]) - q_table[state][action]
        )

        state = next_state
        total_reward += reward

        if done:
            break

    episode_rewards.append(total_reward)
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    if ep % 100 == 0:
        print(f"Episode {ep}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

    if ep % SAVE_EVERY == 0 and ep != 0:
        with open(QTABLE_FILE, "wb") as f:
            pickle.dump(dict(q_table), f)
        with open(EPSILON_FILE, "wb") as f:
            pickle.dump(epsilon, f)
        print(f"💾 Saved progress at episode {ep}")

env.close()

# Final Save
with open(QTABLE_FILE, "wb") as f:
    pickle.dump(dict(q_table), f)
with open(EPSILON_FILE, "wb") as f:
    pickle.dump(epsilon, f)

print("\n✅ Training complete. Q-table and epsilon saved.")

# -----------------------
# Plotting
# -----------------------
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Reward over Time")
plt.grid(True)
plt.tight_layout()
plt.show()
