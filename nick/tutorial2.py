import gymnasium as gym
from stable_baselines3 import A2C,PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv   # or DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import os

MODEL = PPO

N_ENVS = 8              # how many CartPoles you want in parallel
TOTAL_STEPS = 600_000    # total gradient steps across *all* envs
ROLLOUT = 2048    

models_dir = f"nick/model_parameters/{MODEL.__name__}"
print(models_dir)
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

logs_dir = f"nick/model_logs/{MODEL.__name__}"
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

# Initialise the environment
vec_env = make_vec_env(
        "Humanoid-v5",  # or any other env
        n_envs = N_ENVS,
        vec_env_cls = SubprocVecEnv,   # swap for DummyVecEnv if IO-bound
        seed = 0,

    )
model = MODEL(
        "MlpPolicy",
        vec_env,
        device="cuda",            # or "cpu"
        n_steps=ROLLOUT // N_ENVS,
        verbose=1,
    )
# Initialise Model

model.learn(total_timesteps=TOTAL_STEPS,reset_num_timesteps=False)
# for i in range(50):
    # model.save(f"{models_dir}/model_params")

vec_env.close()