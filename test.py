import gym
import simple_driving
import numpy as np
import pickle

# ------------------------
# Configurable Parameters
# ------------------------
NUM_EPISODES = 10
MAX_STEPS = 200
BINS = 20  # Lower if using 4D state
QTABLE_FILE = "qtable.pkl"

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

# ------------------------
# Load Q-table
# ------------------------
with open(QTABLE_FILE, "rb") as f:
    q_table = pickle.load(f)

# ------------------------
# Evaluate
# ------------------------
env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=True, isDiscrete=True)

successes = 0
failures = 0
total_rewards = []
steps_to_goal = []
episode_lengths = []

for ep in range(NUM_EPISODES):
    state, _ = env.reset()
    state = discretize(state)

    ep_reward = 0
    for step in range(MAX_STEPS):
        action = np.argmax(q_table.get(state, np.zeros(9)))
        next_state, reward, done, _, _ = env.step(action)
        next_state = discretize(next_state)

        ep_reward += reward
        state = next_state

        if done:
            if reward > 0:  # Successful goal reached
                successes += 1
                steps_to_goal.append(step + 1)
            else:  # Episode ended without reaching goal
                failures += 1
            break
    else:
        failures += 1  # Max steps reached without success

    episode_lengths.append(step + 1)
    total_rewards.append(ep_reward)

env.close()

# ------------------------
# Results
# ------------------------
print("\n Evaluation Results (Q-Learning Agent)")
print(f"Evaluated {NUM_EPISODES} episodes")
print(f" Success rate: {successes}/{NUM_EPISODES} ({successes / NUM_EPISODES * 100:.1f}%)")
print(f" Failure count: {failures}")
print(f" Average reward: {np.mean(total_rewards):.2f}")
print(f" Reward range: {np.min(total_rewards):.2f} → {np.max(total_rewards):.2f}")
if steps_to_goal:
    print(f" Avg steps to goal (successes): {np.mean(steps_to_goal):.1f}")
else:
    print("  No successful episodes — check training or state encoding.")
print(f"  Avg episode length: {np.mean(episode_lengths):.1f} steps")
