
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np
import os
import time 
from gymnasium.wrappers import TimeLimit

def evaluate_agent(model_path="ppo_rnd_swimmer", vecnorm_path="ppo_rnd_swimmer_vecnormalize.pkl", episodes=10, render=True):
    # Create the environment first
    model = PPO.load(model_path)
    base_env = TimeLimit(gym.make("Swimmer-v4", render_mode="human" if render else None), max_episode_steps=1500)
    env = DummyVecEnv([lambda: base_env])

    # Load VecNormalize stats BEFORE any reset
    if os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False
        print("✅ Loaded VecNormalize statistics.")

    # Load the trained PPO model
    all_rewards = []

    for ep in range(episodes):
        obs = env.reset()
        done = [False]
        t = 0
        total_reward = 0.0

        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            print("Action:", action[0])  # Debug line to check if policy is active
            obs, reward, done, _ = env.step(action)
            total_reward += reward[0]
            t += 1

            if render:
                time.sleep(1 / 60)  # simulate ~60 FPS

        print(f"🎬 Episode {ep + 1} finished. Steps: {t}, Total reward: {total_reward:.2f}")
        all_rewards.append(total_reward)

    env.close()
    avg_reward = np.mean(all_rewards)
    print(f"🏁 Average reward over {episodes} episodes: {avg_reward:.2f}")

if __name__ == "__main__":
    evaluate_agent()
