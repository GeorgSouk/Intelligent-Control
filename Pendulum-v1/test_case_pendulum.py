import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium.wrappers import RecordVideo
import numpy as np
import os

def evaluate_agent(model_path="ppo_rnd_pendulum", vecnorm_path="ppo_rnd_pendulum_vecnormalize.pkl",
                   episodes=10, save_video=True):
    video_dir = "videos"
    os.makedirs(video_dir, exist_ok=True)

    model = PPO.load(model_path)
    base_env = gym.make("Pendulum-v1", render_mode="rgb_array")

    if save_video:
        base_env = RecordVideo(
            base_env,
            video_folder=video_dir,
            episode_trigger=lambda e: True,  
            name_prefix="ppo_rnd_pendulum_demo"
        )

    env = DummyVecEnv([lambda: base_env])

    if os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False
        print("Loaded VecNormalize statistics.")

    all_rewards = []

    for ep in range(episodes):
        obs = env.reset()
        done = [False]
        total_reward = 0.0

        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total_reward += reward[0]

        print(f"Episode {ep + 1}: Total reward = {total_reward:.2f}")
        all_rewards.append(total_reward)

    env.close()
    avg_reward = np.mean(all_rewards)
    print(f"\nAverage reward over {episodes} episodes: {avg_reward:.2f}")
    print(f"Videos saved to: {video_dir}/")

if __name__ == "__main__":
    evaluate_agent()
