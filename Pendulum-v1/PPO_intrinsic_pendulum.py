
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium import Wrapper
from seed_set import SeedSetter

# ---------- RND Network ----------
class RNDModel(nn.Module):
    def __init__(self, input_dim, output_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.model(x)

# ---------- RND Wrapper (Potential-Based Shaping) ----------
class RNDWrapper(Wrapper):
    def __init__(self, env, rnd_target, rnd_predictor, optimizer,
                 initial_beta=0.1, final_beta=0.01, decay_rate=1e-4, gamma=0.99):
        super().__init__(env)
        self.rnd_target = rnd_target
        self.rnd_predictor = rnd_predictor
        self.optimizer = optimizer
        self.decay_rate = decay_rate
        self.initial_beta = initial_beta
        self.final_beta = final_beta
        self.gamma = gamma
        self.episode_count = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.episode_rewards = []
        self.extrinsic_rewards = []
        self.intrinsic_rewards = []
        self.reset_rewards()
        self.last_obs = None  

    def reset_rewards(self):
        self.current_episode_reward = 0.0
        self.current_extrinsic = 0.0
        self.current_intrinsic = 0.0

    def reset(self, *, seed=None, options=None):
        result = self.env.reset(seed=seed, options=options)
        obs, info = result if isinstance(result, tuple) else (result, {})
        self.reset_rewards()
        self.episode_count += 1
        self.last_obs = obs  
        return obs, info

    def step(self, action):
        obs_tensor = torch.tensor(self.last_obs, dtype=torch.float32).unsqueeze(0).to(self.device)  
        with torch.no_grad():
            target_feature_s = self.rnd_target(obs_tensor)
        predicted_feature_s = self.rnd_predictor(obs_tensor)
        Phi_s = torch.mean((predicted_feature_s - target_feature_s) ** 2)

        obs_next, extrinsic_reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        obs_next_tensor = torch.tensor(obs_next, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            target_feature_s_next = self.rnd_target(obs_next_tensor)
        predicted_feature_s_next = self.rnd_predictor(obs_next_tensor)
        Phi_s_next = torch.mean((predicted_feature_s_next - target_feature_s_next) ** 2)

        shaping_reward = self.gamma * Phi_s_next.item() - Phi_s.item()

        loss = Phi_s
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        beta = self.final_beta + (self.initial_beta - self.final_beta) * np.exp(-self.decay_rate * self.episode_count)
        total_reward = extrinsic_reward + beta * shaping_reward

        self.current_episode_reward += total_reward
        self.current_extrinsic += extrinsic_reward
        self.current_intrinsic += shaping_reward

        self.last_obs = obs_next  

        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.extrinsic_rewards.append(self.current_extrinsic)
            self.intrinsic_rewards.append(self.current_intrinsic)

        return obs_next, total_reward, terminated, truncated, info

# ---------- Plotting ----------
def plot_rewards(total, extrinsic, intrinsic):
    plt.figure(figsize=(10, 5))
    plt.plot(total, label="Total Reward")
    plt.plot(extrinsic, label="Extrinsic Reward")
    plt.plot(intrinsic, label="Intrinsic (Shaping) Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Progress with Potential-Based RND")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ---------- Main Training ----------
def main():
    seed = 42
    seeder = SeedSetter(seed)
    env = gym.make("Pendulum-v1")
    seeder.apply_to_env(env)
    obs_dim = env.observation_space.shape[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rnd_target = RNDModel(obs_dim).to(device)
    rnd_predictor = RNDModel(obs_dim).to(device)
    for p in rnd_target.parameters():
        p.requires_grad = False

    optimizer = optim.Adam(rnd_predictor.parameters(), lr=1e-4)

    base_env = RNDWrapper(env, rnd_target, rnd_predictor, optimizer, gamma=0.99)
    vec_env = DummyVecEnv([lambda: base_env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    print("Training PPO with Potential-Based RND")
    model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="./ppo_rnd_tensorboard/",
                learning_rate=6e-4, n_steps=2048, batch_size=64, n_epochs=10)
    model.learn(total_timesteps=100_000)

    model.save("ppo_rnd_pendulum")
    vec_env.save("ppo_rnd_pendulum_vecnormalize.pkl")
    print("Plotting results...")
    plot_rewards(base_env.episode_rewards, base_env.extrinsic_rewards, base_env.intrinsic_rewards)
    os.makedirs("logs", exist_ok=True)
    np.save("logs/ppo_rnd_total_pendulum_rewards.npy", np.array(base_env.episode_rewards))
    np.save("logs/ppo_rnd_extrinsic_pendulum_rewards.npy", np.array(base_env.extrinsic_rewards))
    np.save("logs/ppo_rnd_intrinsic_pendulum_rewards.npy", np.array(base_env.intrinsic_rewards))
    print("Training complete. Saved model and reward logs.")

if __name__ == "__main__":
    main()
