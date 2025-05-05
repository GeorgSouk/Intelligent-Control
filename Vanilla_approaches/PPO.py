import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ---------- Noisy Observation Wrapper ----------
class NoisyObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, noise_scale=5.0):
        super().__init__(env)
        self.noise_scale = noise_scale

    def observation(self, obs):
        noise = np.random.normal(0, self.noise_scale, size=obs.shape)
        return obs + noise

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
    plt.title("Training Progress - PPO Baseline (Noisy Observations)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ---------- Main Training ----------
def main():
    env = gym.make("Pendulum-v1")
    env = NoisyObsWrapper(env, noise_scale=5.0)  # Match RND noise scale
    monitored_env = RewardMonitor(env)

    vec_env = DummyVecEnv([lambda: monitored_env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    print("Training PPO Baseline with Noisy Observations...")
    model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="./ppo_noisy_tensorboard/",
                learning_rate=6e-4, n_steps=2048, batch_size=64, n_epochs=10)
    model.learn(total_timesteps=50_000)

    model.save("ppo_noisy_baseline")

    # Get reward logs (excluding first dummy reward on reset)
    rewards = monitored_env.episode_rewards[1:]

    print("Plotting results...")
    plot_rewards(rewards)

    print("Saving reward logs...")
    os.makedirs("logs", exist_ok=True)
    np.save("logs/ppo_noisy_baseline_rewards.npy", np.array(rewards))
    print("Saved to logs/ppo_noisy_baseline_rewards.npy")

if __name__ == "__main__":
    main()
