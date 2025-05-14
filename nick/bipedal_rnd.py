import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from rnd_reward_wrapper import RNDRewardWrapper  # uses the new MLP networks

N_ENVS = 16
BASE_ENV = "CartPole-v1"           # any low-dim env; no pixels needed

vec_env = make_vec_env(
    "Humanoid-v5",
    n_envs=8,
    wrapper_class=RNDRewardWrapper,                      # <- add wrapper here
    wrapper_kwargs=dict(beta=0.5, lr=1e-4),             # extra args for RND
    vec_env_cls=SubprocVecEnv,                           # or DummyVecEnv
    seed=0,
)
vec_env = RNDRewardWrapper(vec_env, beta=0.5, lr=1e-4)  # curiosity on

model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    n_steps=2048 // N_ENVS,
    device="cuda"
)
model.learn(total_timesteps=500_000)
