import numpy as np
import matplotlib.pyplot as plt
import os

def smooth(data, window=10):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')

def load_shared_rewards():
    logs_dir = "logs"
    rewards = {}

    paths = {
        "PPO + RND": "ppo_rnd_total_swimmer_rewards.npy",
        "DDPG": "ddpg_swimmer_rewards.npy",
    }

    for label, file_name in paths.items():
        path = os.path.join(logs_dir, file_name)
        if os.path.exists(path):
            rewards[label] = np.load(path)
        else:
            print(f"Warning: {path} not found. Skipping '{label}'.")

    return rewards

def plot_all_rewards_shared(rewards_dict):
    plt.figure(figsize=(14, 7))
    for label, rewards in rewards_dict.items():
        plt.plot(smooth(rewards), label=label)
    plt.xlabel("Episode")
    plt.ylabel("Smoothed Total Reward")
    plt.title("Comparison of RL Algorithms on Noisy Pendulum-v1")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    rewards_data = load_shared_rewards()
    plot_all_rewards_shared(rewards_data)
