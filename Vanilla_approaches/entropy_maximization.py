import gym
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from seed_set import SeedSetter

class EntropyQAgent:
    def __init__(self, state_dim, n_actions, lr=0.1, discount=0.99, eps=1.0,
                 eps_min=0.01, eps_decay=0.995, entropy_wt=0.01):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.q = np.zeros((state_dim, n_actions))
        self.lr = lr
        self.discount = discount
        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.entropy_wt = entropy_wt

    def action(self, s):
        if random.random() < self.eps:
            return random.choice(range(0,self.n_actions))
        return int(np.argmax(self.q[s]))  

    def _softmax(self, values):
        max_val = np.max(values)
        exp_vals = np.exp(values - max_val)
        probs = exp_vals / np.sum(exp_vals)
        return probs

    def _entropy(self, p):
        return -np.sum(p * np.log(p + 1e-10))  

    def update(self, s, a, r, s_next, done):
        q_next = self.q[s_next]
        probs = self._softmax(q_next)
        H = self._entropy(probs)

        target = r + self.discount * (np.max(q_next) + self.entropy_wt * H) * (0 if done else 1)
        delta = target - self.q[s][a]
        self.q[s][a] += self.lr * delta

        if done:
            self.eps = max(self.eps * self.eps_decay, self.eps_min)

    def dump_q(self):
        return self.q

# ---------- Noisy Observation Wrapper ----------
class NoisyObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, noise_scale=5.0):
        super().__init__(env)
        self.noise_scale = noise_scale

    def observation(self, obs):
        noise = np.random.normal(0, self.noise_scale, size=obs.shape)
        return obs + noise 

# ---------- Discretization ----------
def create_bins():
    obs_bins = [
        np.linspace(-1.0, 1.0, 10),   # cos(theta)
        np.linspace(-1.0, 1.0, 10),   # sin(theta)
        np.linspace(-8.0, 8.0, 10)    # theta_dot
    ]
    action_bins = np.linspace(-2.0, 2.0, 5)  # Discretized actions
    return obs_bins, action_bins

def discretize_obs(obs, obs_bins):
    return tuple(int(np.digitize(o, b) - 1) for o, b in zip(obs, obs_bins))

def flatten_state(state_idx, shape):
    return state_idx[0] * shape[1] * shape[2] + state_idx[1] * shape[2] + state_idx[2]

# ---------- Training Loop ----------
def train_entropy_q_learning(episodes=250):
    seed = 42
    seeder = SeedSetter(seed)

    env = gym.make("Pendulum-v1")
    seeder.apply_to_env(env)    
    env = NoisyObsWrapper(env, noise_scale=5.0)
    obs_bins, action_bins = create_bins()

    state_shape = tuple(len(b) for b in obs_bins)
    n_states = np.prod(state_shape)
    n_actions = len(action_bins)

    agent = EntropyQAgent(state_dim=n_states, n_actions=n_actions)

    rewards = []

    for ep in range(episodes):
        obs = env.reset()[0]
        s_idx = flatten_state(discretize_obs(obs, obs_bins), state_shape)
        total_reward = 0
        done = False

        while not done:
            a = agent.action(s_idx)
            action_val = [action_bins[a]]
            next_obs, reward, terminated, truncated, _ = env.step(action_val)
            done = terminated or truncated

            s_next_idx = flatten_state(discretize_obs(next_obs, obs_bins), state_shape)
            agent.update(s_idx, a, reward, s_next_idx, done)
            s_idx = s_next_idx
            total_reward += reward

        rewards.append(total_reward)

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}, Reward: {total_reward:.2f}, Epsilon: {agent.eps:.3f}")

    env.close()
    return rewards

# ---------- Main ----------
def main():
    rewards = train_entropy_q_learning(episodes=250)

    os.makedirs("logs", exist_ok=True)
    np.save("logs/entropy_q_learning_noisy_pendulum_rewards.npy", np.array(rewards))
    print("Saved to logs/entropy_q_learning_noisy_pendulum_rewards.npy")

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Entropy Q-learning (Noisy Pendulum)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Entropy-augmented Q-learning on Noisy Pendulum-v1")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
