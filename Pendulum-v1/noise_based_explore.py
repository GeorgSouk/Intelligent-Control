import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
import matplotlib.pyplot as plt
from collections import deque
from seed_set import SeedSetter

# --------------------- Replay Buffer ---------------------
class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*samples))
        return (
            torch.FloatTensor(states),
            torch.FloatTensor(actions),
            torch.FloatTensor(rewards).unsqueeze(1),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)

# --------------------- Actor and Critic ---------------------
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))

# --------------------- DDPG Agent ---------------------
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.max_action = max_action
        self.replay_buffer = ReplayBuffer()
        self.gamma = 0.99
        self.tau = 0.005

    def add_parameter_noise(self, stddev=0.5):
        for param, param_target in zip(self.actor.parameters(), self.actor_target.parameters()):
            if param.requires_grad:
                noise = torch.normal(0, stddev, size=param.data.size())
                param.data.copy_(param_target.data + noise)

    def select_action(self, state, explore=True):
        state_tensor = torch.FloatTensor(state.reshape(1, -1))
        if explore:
            self.add_parameter_noise(stddev=0.1)
        action = self.actor(state_tensor).detach().numpy()[0]
        return np.clip(action, -self.max_action, self.max_action)

    def train(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_Q = self.critic_target(next_states, next_actions)
            target_Q = rewards + (1 - dones) * self.gamma * target_Q

        current_Q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# --------------------- Noisy Observation Wrapper ---------------------
class NoisyObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, noise_scale=0.5):
        super().__init__(env)
        self.noise_scale = noise_scale

    def observation(self, obs):
        noise = np.random.normal(0, self.noise_scale, size=obs.shape)
        return obs + noise

# --------------------- Main Training ---------------------
def main():
    seed = 42
    seeder = SeedSetter(seed)

    env = gym.make("Pendulum-v1")
    seeder.apply_to_env(env)
    env = NoisyObsWrapper(env, noise_scale=0.5)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = DDPGAgent(state_dim, action_dim, max_action)

    episodes = 250
    reward_log = []

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.push(state, action, reward, next_state, float(done))
            state = next_state
            total_reward += reward
            agent.train()

        reward_log.append(total_reward)

        if (ep + 1) % 10 == 0:
            print(f"Episode {ep + 1}, Total Reward: {total_reward:.2f}")

    os.makedirs("logs", exist_ok=True)
    np.save("logs/ddpg_noisy_pendulum_rewards.npy", np.array(reward_log))
    print("Saved to logs/ddpg_noisy_pendulum_rewards.npy")

    plt.plot(reward_log)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DDPG with Noisy Observations - Reward Curve")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
