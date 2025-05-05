import gym
import numpy as np
import matplotlib.pyplot as plt
import os

# ---------- Noisy Observation Wrapper ----------
class NoisyObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, noise_scale=5.0):
        super().__init__(env)
        self.noise_scale = noise_scale

    def observation(self, obs):
        noise = np.random.normal(0, self.noise_scale, size=obs.shape)
        return obs + noise

# ---------- Q-learning on Noisy Pendulum-v1 ----------
def run_q_learning_noisy_pendulum(episodes=250):
    env = gym.make("Pendulum-v1")
    env = NoisyObsWrapper(env, noise_scale=5.0)

    action_bins = np.linspace(-2.0, 2.0, 5)  # Discretized actions
    n_actions = len(action_bins)

    obs_bins = [
        np.linspace(-1.0, 1.0, 10),   # cos(theta)
        np.linspace(-1.0, 1.0, 10),   # sin(theta)
        np.linspace(-8.0, 8.0, 10)    # theta_dot
    ]

    def discretize_obs(obs):
        return tuple(int(np.digitize(o, b) - 1) for o, b in zip(obs, obs_bins))

    q_table = np.zeros([10, 10, 10, n_actions])
    alpha = 0.1
    gamma = 0.99
    epsilon = 1.0
    decay = 0.995

    rewards = []

    for ep in range(episodes):
        obs = env.reset()[0]
        state = discretize_obs(obs)
        total_reward = 0
        done = False

        while not done:
            if np.random.rand() < epsilon:
                action_idx = np.random.randint(n_actions)
            else:
                action_idx = np.argmax(q_table[state])
            action = [action_bins[action_idx]]
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = discretize_obs(next_obs)
            done = terminated or truncated

            q_table[state][action_idx] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state][action_idx])
            state = next_state
            total_reward += reward

        rewards.append(total_reward)
        epsilon *= decay

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

    env.close()
    return rewards

# ---------- Plotting ----------
def plot_rewards(rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Q-learning (Noisy Pendulum)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Q-learning with Discretized Noisy Pendulum-v1")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---------- Main ----------
def main():
    print("Running Q-learning on Noisy Pendulum-v1...")
    rewards = run_q_learning_noisy_pendulum(episodes=250)

    os.makedirs("logs", exist_ok=True)
    np.save("logs/q_learning_noisy_pendulum_rewards.npy", np.array(rewards))
    print("Saved to logs/q_learning_noisy_pendulum_rewards.npy")

    plot_rewards(rewards)

if __name__ == "__main__":
    main()
