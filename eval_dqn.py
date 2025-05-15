import torch
import torch.nn as nn
import numpy as np
import gym
import simple_driving
import time

QMODEL_FILE = "dqn_model.pth"

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

def evaluate_model(episodes=10, render=True):
    env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=render, isDiscrete=True)
    state, _ = env.reset()
    obs_dim = len(state)
    print("Initial state:", state, "Length:", obs_dim)
    n_actions = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQN(obs_dim, n_actions).to(device)
    model.load_state_dict(torch.load(QMODEL_FILE, map_location=device))
    model.eval()

    rewards = []
    successes = 0
    failures = 0
    episode_lengths = []
    steps_to_goal = []

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False
        reached_goal = False  # local flag

        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = model(state_tensor)
                action = torch.argmax(q_values).item()

            state, reward, done, _, _ = env.step(action)
            total_reward += reward
            steps += 1

            if render:
                time.sleep(0.02)

            # 👇 Check goal after each step
            if done and reward >= 50:
                reached_goal = True

        rewards.append(total_reward)
        episode_lengths.append(steps)

        if reached_goal:
            successes += 1
            steps_to_goal.append(steps)
        else:
            failures += 1

        print(f"Episode {ep + 1}: Reward = {total_reward:.2f} | Steps = {steps} | Success = {reached_goal}")

    # Metrics
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    min_reward = np.min(rewards)
    max_reward = np.max(rewards)
    success_rate = successes / episodes * 100
    avg_steps = np.mean(episode_lengths)

    print("\n📊 Evaluation Summary:")
    print(f"✅ Success rate: {success_rate:.2f}% ({successes}/{episodes})")
    print(f"🏆 Avg reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"🔻 Min reward: {min_reward:.2f} | 🔺 Max reward: {max_reward:.2f}")
    if steps_to_goal:
        print(f"🏁 Avg steps to goal (successes only): {np.mean(steps_to_goal):.1f}")
    else:
        print("⚠️  No successful episodes — check training or environment setup.")
    print(f"📏 Avg episode length: {avg_steps:.2f} steps")

    env.close()


if __name__ == "__main__":
    evaluate_model(episodes=10, render=True)
