import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from seed_set import SeedSetter
from gymnasium.wrappers import TimeLimit

# ---------- Reward Tracking Wrapper ----------
class RewardMonitor(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.episode_rewards = []
        self.current_reward = 0.0

    def reset(self, *, seed=None, options=None):
        if hasattr(self, 'current_reward'):
            self.episode_rewards.append(self.current_reward)
        self.current_reward = 0.0
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_reward += reward
        return obs, reward, terminated, truncated, info

# ---------- Plotting ----------
def plot_rewards(reward_list):
    plt.figure(figsize=(10, 5))
    plt.plot(reward_list, label="Total Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Progress - PPO on Swimmer")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ---------- Main Training ----------
def main():
    seed = 42
    seeder = SeedSetter(seed)
    env = TimeLimit(gym.make("Swimmer-v4"), max_episode_steps=1500)
    seeder.apply_to_env(env)

    monitored_env = RewardMonitor(env)

    vec_env = DummyVecEnv([lambda: monitored_env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    print("Training PPO on Swimmer-v4...")
    model = PPO("MlpPolicy", vec_env, verbose=1, learning_rate=6e-4, n_steps=4096, batch_size=256, n_epochs=20)
    model.learn(total_timesteps=500_000)

    model.save("ppo_swimmer")
    rewards = monitored_env.episode_rewards[1:]

    print("Plotting results...")
    plot_rewards(rewards)

    print("Saving reward logs...")
    os.makedirs("logs", exist_ok=True)
    np.save("logs/ppo_swimmer_rewards.npy", np.array(rewards))

if __name__ == "__main__":
    main()