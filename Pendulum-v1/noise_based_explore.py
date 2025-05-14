"""
ddpg_gpu_noisy_pendulum.py

A GPU-enabled implementation of the Deep Deterministic Policy Gradient (DDPG) agent
with noisy observations for OpenAI Gymnasium environments.

Features
--------
- Automatic device selection (CUDA GPU if available, otherwise CPU)
- Replay buffer for experience storage
- Actor and Critic networks with target counterparts
- Parameter noise for exploration
- Noisy observation wrapper
- Training loop with reward logging and plotting
- Reproducible seeding across Python, NumPy, PyTorch, and Gymnasium

Usage
-----
>>> python ddpg_gpu_noisy_pendulum.py

Requirements
------------
- gymnasium
- numpy
- torch
- matplotlib
- seed_set.py (SeedSetter utility module)
"""
import os
import random
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
from seed_set import SeedSetter

# --------------------- Device Configuration ---------------------
# Automatically choose GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------- Replay Buffer ---------------------
class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples (s, a, r, s', done).
    """
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        """Add a transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Sample a batch of transitions and convert to PyTorch tensors on the chosen device.
        Returns:
            states: (batch_size, state_dim)
            actions: (batch_size, action_dim)
            rewards: (batch_size, 1)
            next_states: (batch_size, state_dim)
            dones: (batch_size, 1)
        """
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*samples))

        # Convert to FloatTensors and move to device
        return (
            torch.FloatTensor(states).to(device),
            torch.FloatTensor(actions).to(device),
            torch.FloatTensor(rewards).unsqueeze(1).to(device),
            torch.FloatTensor(next_states).to(device),
            torch.FloatTensor(dones).unsqueeze(1).to(device)
        )

    def __len__(self):
        return len(self.buffer)

# --------------------- Actor and Critic Networks ---------------------
class Actor(nn.Module):
    """
    Policy network that maps states to actions in [-max_action, max_action].
    """
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
        """Compute action for a given state batch."""
        return self.max_action * self.net(state)

class Critic(nn.Module):
    """
    Q-value network that maps state-action pairs to expected return.
    """
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
        """Compute Q-values for state-action pairs."""
        return self.net(torch.cat([state, action], dim=1))

# --------------------- DDPG Agent ---------------------
class DDPGAgent:
    """
    Deep Deterministic Policy Gradient agent with target networks and parameter noise.
    """
    def __init__(self, state_dim, action_dim, max_action):
        # Instantiate networks
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers for actor and critic
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        # Experience replay and hyperparameters
        self.replay_buffer = ReplayBuffer()
        self.gamma = 0.99  # Discount factor
        self.tau = 0.005   # Soft update factor

    def add_parameter_noise(self, stddev=0.5,delta = 0.6,alpha = 1.01):
        """Apply parameter noise to the actor target network for exploration."""
        for param, param_target in zip(self.actor.parameters(), self.actor_target.parameters()):
            if param.requires_grad:
                noise = torch.normal(0, stddev, size=param.data.size(), device=device)
                d = torch.norm(noise).item()
                if d > delta:
                    noise = noise / alpha
                else:
                    noise = noise * alpha
                # print(f"alpha {alpha}, delta {delta}, d {d}")
                param.data.copy_(param_target.data + noise)

    def select_action(self, state, explore=True):
        """
        Choose an action given the current policy, optionally adding parameter noise.
        Args:
            state: numpy array of shape (state_dim,)
            explore: whether to perturb parameters for exploration
        Returns:
            clipped action as numpy array
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        if explore:
            self.add_parameter_noise(stddev=0.01)
        action = self.actor(state_tensor).cpu().detach().numpy()[0]
        return np.clip(action, -self.actor.max_action, self.actor.max_action)

    def train(self, batch_size=64):
        """
        Sample a batch from replay buffer and update the actor and critic networks.
        """
        if len(self.replay_buffer) < batch_size:
            return

        # Sample experiences
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # Compute target Q-values using target networks
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_Q = self.critic_target(next_states, next_actions)
            target_Q = rewards + (1 - dones) * self.gamma * target_Q

        # Critic update (MSE loss)
        current_Q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_Q, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update (policy gradient)
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update of target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# --------------------- Noisy Observation Wrapper ---------------------
class NoisyObsWrapper(gym.ObservationWrapper):
    """
    Gymnasium wrapper that adds Gaussian noise to observations for robustness testing.
    """
    def __init__(self, env, noise_scale=0.1):
        super().__init__(env)
        self.noise_scale = noise_scale

    def observation(self, obs):
        """Add Gaussian noise to each observation element."""
        noise = np.random.normal(0, self.noise_scale, size=obs.shape)
        return obs + noise

# --------------------- Main Training Loop ---------------------
def main():
    # Set seeds for reproducibility
    seed = 42
    seeder = SeedSetter(seed)

    # Initialize environment with noisy observations
    env = gym.make("Pendulum-v1")
    seeder.apply_to_env(env)
    env = NoisyObsWrapper(env, noise_scale=0.01)

    # Extract environment dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Instantiate DDPG agent
    agent = DDPGAgent(state_dim, action_dim, max_action)

    # Training parameters
    episodes = 250
    reward_log = []

    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Select and perform an action
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store transition and train
            agent.replay_buffer.push(state, action, reward, next_state, float(done))
            state = next_state
            total_reward += reward
            agent.train()

        reward_log.append(total_reward)

        # Log progress every 10 episodes
        if ep % 10 == 0:
            print(f"Episode {ep}, Total Reward: {total_reward:.2f}")

    # Create logs directory and save rewards
    os.makedirs("logs", exist_ok=True)
    np.save("logs/ddpg_noisy_pendulum_rewards.npy", np.array(reward_log))
    print("Saved reward history to logs/ddpg_noisy_pendulum_rewards.npy")

    # Plot the learning curve
    plt.plot(reward_log)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DDPG with Noisy Observations - Reward Curve")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
