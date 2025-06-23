import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_checker import check_env
import os
import matplotlib.pyplot as plt

class Autoencoder(nn.Module):
	def __init__(self, input_dim=6, latent_dim=1):
		super().__init__()
		# Encoder: s_t + s_t+1 →  latent representation z (i think its the torque) 
		self.encoder = nn.Sequential(
			nn.Linear(input_dim, 128),
			nn.ReLU(),
			nn.Linear(128, 128),
			nn.ReLU(),
			nn.Linear(128, latent_dim),
		)
		# Decoder: z → approximation s_t+1 
		self.decoder = nn.Sequential(
			nn.Linear(latent_dim, 128),
			nn.ReLU(),
   			nn.Linear(128, 128),
			nn.ReLU(),
			nn.Linear(128, 3),
			nn.Sigmoid(),              
		)

	def forward(self, x):
		
		z = self.encoder(x)
		x_hat = self.decoder(z)
		return x_hat, z

def vae_loss(s, s_next, a, encoder, decoder, prior_net, expert_policy, beta=1.0):
    # s: current state batch, s_next: next state batch, a: actions
    # encoder returns mu_z, logvar_z
    mu, logvar = encoder(s, s_next)
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    z = mu + eps * std  # reparameterization

    # decoder reconstructs next state
    s_next_hat = decoder(s, z)
    recon_loss = F.mse_loss(s_next_hat, s_next, reduction='none')
    recon_loss = recon_loss.view(recon_loss.size(0), -1).sum(dim=1).mean()

    # KL(q(z|s,s') || p(z|s))
    mu0, logvar0 = prior_net(s)
    kl1 = 0.5 * (
        logvar0 - logvar - 1 +
        (torch.exp(logvar) + (mu - mu0).pow(2)) / torch.exp(logvar0)
    ).sum(dim=1).mean()

    # KL(q(z|s,s') || expert policy pi_E(a|s))
    # assume expert_policy returns logits over actions
    z_scaled = z  # if any transform needed
    log_q = -0.5 * (logvar + (z - mu).pow(2) / torch.exp(logvar)).sum(dim=1)
    log_pi = expert_policy.log_prob(a, given_state=s)        # implement accordingly
    kl2 = (log_q - log_pi).mean()

    loss = -recon_loss + kl1 + beta * kl2
    return loss, recon_loss, kl1, kl2

class GirmEnv(gym.Wrapper):
	def __init__(self, env):
		# Call the parent constructor, so we can access self.env later
		super().__init__(env)
		self.action_space = env.action_space
		self.observation_space = env.observation_space
		self.reward_history = np.array([])
	def reset(self, **kwargs):

		self.current_step = 0
		return self.env.reset(**kwargs)

	def step(self, action):
		
		self.current_step += 1
		obs, reward, terminated, truncated, info = self.env.step(action)
		# modified reward 
		self.reward_history = np.append(self.reward_history, reward)
		# print(self.reward_history)
		# Overwrite the truncation signal when when the number of steps reaches the maximum

		return obs, reward, terminated, truncated, info
def main():
	base_env = gym.make("Pendulum-v1", render_mode = "rgb_array")
	base_env = GirmEnv(base_env)

	vec_env = DummyVecEnv([lambda: base_env])
	# vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
	check_env(base_env)
	
	save_dir = "./nick/tmp/grim_net/"
	os.makedirs(save_dir, exist_ok=True)
	expert_model = PPO.load(f"{save_dir}/PPO_master", env=vec_env, device="cpu")
 
	
	model = PPO("MlpPolicy",vec_env, verbose=0,device="cpu")

	
	vec_env = model.get_env()
	obs = vec_env.reset() 
 
	model.save(f"{save_dir}/PPO_girm")

 
	print("Training done")
	for i in range(2000):
		
		action, _state = model.predict(obs,deterministic = True)
		obs, reward, done, info = vec_env.step(action)
		vec_env.render("human")
		if done:
			obs = vec_env.reset()
	vec_env.close()

	# Plot the reward history from the wrapper environment
	plt.figure()
	# print(base_env.reward_history)
	smoothed_rewards = np.convolve(base_env.reward_history, np.ones(1000)/1000, mode='valid')
	np.save("smoothed_reward_history.npy", smoothed_rewards)
	plt.plot(base_env.reward_history)
	plt.xlabel("Step")
	plt.ylabel("Reward")
	plt.title("Reward History")
	plt.show()
if __name__ == "__main__":
	main()