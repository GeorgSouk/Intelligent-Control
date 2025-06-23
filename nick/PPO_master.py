import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os
import csv
def fill_Buffer():
	env = gym.make("Pendulum-v1", render_mode = "rgb_array")
	env = DummyVecEnv([lambda: env])
		
	load_dir = "./nick/tmp/master_net/"
	model = PPO.load(f"{load_dir}/PPO_master", env=env, device="cpu")
	save_file = "./nick/tmp/master_parameters.csv"

	vec_env = model.get_env()
	obs = vec_env.reset() 
	# Save all the s_t,a_t,s_t+1 to the save file

	buffer = []

	prev_obs = obs.copy()
	N = 300_000
	for i in range(N):
		action, _state = model.predict(prev_obs, deterministic=True)
		next_obs, reward, done, info = env.step(action)
		# Flatten observations and actions for saving
		buffer.append(np.concatenate([prev_obs.flatten(), action.flatten(), next_obs.flatten()]))
		prev_obs = next_obs
		if done:
			prev_obs = vec_env.reset()

	# Save buffer to CSV
	with open(save_file, "w", newline="") as f:
		writer = csv.writer(f)
		# Write header
		obs_dim = obs.size
		act_dim = action.size
		header = [f"s_t_{i}" for i in range(obs_dim)] + [f"a_t_{i}" for i in range(act_dim)] + [f"s_t+1_{i}" for i in range(obs_dim)]
		writer.writerow(header)
		writer.writerows(buffer)
	# for i in range(1000):
		
	# 	action, _state = model.predict(obs,deterministic = True)
	# 	obs, reward, done, info = env.step(action)
	# 	vec_env.render("human")
	# 	if done:
	# 		obs = vec_env.reset()
	env.close()

def main():
	env = gym.make("Pendulum-v1", render_mode = "rgb_array")
	env = DummyVecEnv([lambda: env])
	
	# model = PPO("MlpPolicy",env, verbose=1,device="cpu").learn(total_timesteps=350_000)
	
	save_dir = "./nick/tmp/master_net/"
	os.makedirs(save_dir, exist_ok=True)
	# Optionally, load a previously saved model
	# if os.path.exists(f"{save_dir}/PPO_master.zip"):
	model = PPO.load(f"{save_dir}/PPO_master", env=env, device="cpu")
	# model.save(f"{save_dir}/PPO_master")
	
	vec_env = model.get_env()
	obs = vec_env.reset() 
	print("Training done")
	for i in range(1000):
		
		action, _state = model.predict(obs,deterministic = True)
		obs, reward, done, info = env.step(action)
		vec_env.render("human")
		if done:
			obs = vec_env.reset()
	env.close()
if __name__ == "__main__":
	fill_Buffer()