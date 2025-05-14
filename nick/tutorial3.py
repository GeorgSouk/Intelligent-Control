import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make('CartPole-v1',render_mode = "human")  # continuous: LunarLanderContinuous-v2
env.reset()

MODEL = PPO
model = MODEL('MlpPolicy', env, verbose=1, device="cpu")

model_path = f"nick/model_parameters/{MODEL.__name__}/model_params.zip"
model = MODEL.load(model_path, env=env)
print("loaded")
episodes = 5

for ep in range(episodes):
    obs,info = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        action, _ = model.predict(obs)
