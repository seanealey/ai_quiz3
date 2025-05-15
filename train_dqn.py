import torch
import torch.nn as nn
import torch.optim as optim
import gym
import simple_driving
import numpy as np
import random
import os
from collections import deque
import matplotlib.pyplot as plt
import pickle

# -----------------------
# Hyperparameters
# -----------------------
EPISODES = 10000
MAX_STEPS = 200
GAMMA = 0.99
ALPHA = 0.001
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.05
EPSILON_START = 1.0
MEMORY_SIZE = 50000
BATCH_SIZE = 64
SAVE_EVERY = 1000
QMODEL_FILE = "dqn_model.pth"
EPSILON_FILE = "dqn_epsilon.pkl"

# -----------------------
# Biased Exploration
# -----------------------
def biased_exploration():
    probs = [0.05, 0.05, 0.05, 0.10, 0.01, 0.10, 0.20, 0.30, 0.14]
    return np.random.choice(np.arange(9), p=probs)

# -----------------------
# Q-Network
# -----------------------
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# -----------------------
# Replay Buffer
# -----------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        return map(np.array, zip(*batch))

    def __len__(self):
        return len(self.buffer)

# -----------------------
# Environment Setup
# -----------------------
env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=False, isDiscrete=True)
state, _ = env.reset()
obs_dim = len(state)  # ✅ True shape of state
print("Initial state:", state, "Length:", obs_dim)

n_actions = env.action_space.n


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(obs_dim, n_actions).to(device)
target_net = DQN(obs_dim, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=ALPHA)
replay_buffer = ReplayBuffer(MEMORY_SIZE)

# Load epsilon if exists
epsilon = EPSILON_START
if os.path.exists(EPSILON_FILE):
    with open(EPSILON_FILE, "rb") as f:
        epsilon = pickle.load(f)

episode_rewards = []

# -----------------------
# Training Loop
# -----------------------
for ep in range(EPISODES):
    state, _ = env.reset()
    total_reward = 0

    for t in range(MAX_STEPS):
        # Action selection
        if random.random() < epsilon:
            action = biased_exploration()
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = policy_net(state_tensor)
                action = torch.argmax(q_values).item()

        # Take action
        next_state, reward, done, _, _ = env.step(action)
        replay_buffer.push((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        if done:
            break

        # Learn
        if len(replay_buffer) >= BATCH_SIZE:
            states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

            states = torch.FloatTensor(states).to(device)
            actions = torch.LongTensor(actions).unsqueeze(1).to(device)
            rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
            next_states = torch.FloatTensor(next_states).to(device)
            dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

            q_values = policy_net(states).gather(1, actions)
            next_q_values = target_net(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards + GAMMA * next_q_values * (1 - dones)

            loss = nn.MSELoss()(q_values, target_q_values)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # End of episode
    episode_rewards.append(total_reward)
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    if ep % 100 == 0:
        print(f"Episode {ep}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

    if ep % SAVE_EVERY == 0 and ep != 0:
        torch.save(policy_net.state_dict(), QMODEL_FILE)
        with open(EPSILON_FILE, "wb") as f:
            pickle.dump(epsilon, f)
        target_net.load_state_dict(policy_net.state_dict())
        print(f"💾 Model saved at episode {ep}")

# -----------------------
# Final Save
# -----------------------
torch.save(policy_net.state_dict(), QMODEL_FILE)
with open(EPSILON_FILE, "wb") as f:
    pickle.dump(epsilon, f)
print("\n✅ DQN Training complete. Model and epsilon saved.")

# -----------------------
# Plotting
# -----------------------
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("DQN Training Reward over Time")
plt.grid(True)
plt.tight_layout()
plt.show()
