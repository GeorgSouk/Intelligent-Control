import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv   # or DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

N_ENVS = 8              # how many CartPoles you want in parallel
TOTAL_STEPS = 600_000    # total gradient steps across *all* envs
ROLLOUT = 2048           # PPO default – keep n_steps*n_envs ≈ 2 k

if __name__ == "__main__":      # required for Subproc on Windows
    vec_env = make_vec_env(
        "Humanoid-v5",  # or any other env
        n_envs = N_ENVS,
        vec_env_cls=SubprocVecEnv,   # swap for DummyVecEnv if IO-bound
        seed=0,
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        device="cuda",            # or "cpu"
        n_steps=ROLLOUT // N_ENVS,
        verbose=1,
    )
    model.learn(total_timesteps=TOTAL_STEPS)

    # quick evaluation — SB3 divides episodes across the workers
    mean_r, std_r = evaluate_policy(
        model, vec_env, n_eval_episodes=10, deterministic=True
    )
    print(f"mean return = {mean_r:.1f} ± {std_r:.1f}")
    # Run a single episode
    obs = vec_env.reset()                     # array (n_envs, obs_dim)
    dones = np.zeros(vec_env.num_envs, bool)  # per-worker flags

    while not dones.all():                    # stop when *every* worker done
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = vec_env.step(action)
        vec_env.render("human")

    vec_env.close()
    
