import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np
import os

def evaluate_agent(model_path="ppo_rnd_pendulum", vecnorm_path="ppo_rnd_pendulum_vecnormalize.pkl", episodes=20, render=True):
    env = gym.make("Pendulum-v1", render_mode="human" if render else None)
    env = DummyVecEnv([lambda: env])

    if os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False
        print("✅ Loaded VecNormalize statistics.")

    model = PPO.load(model_path)

    all_rewards = []

    for ep in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total_reward += reward[0]  # reward is vectorized

        print(f"Episode {ep + 1}: Total reward = {total_reward:.2f}")
        all_rewards.append(total_reward)

    env.close()
    avg_reward = np.mean(all_rewards)
    print(f"\n🎯 Average reward over {episodes} episodes: {avg_reward:.2f}")

if __name__ == "__main__":
    evaluate_agent()
