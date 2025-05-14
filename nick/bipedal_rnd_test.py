"""
Evaluate a trained MODEL agent on the Humanoid-v4 MuJoCo task.

• Finds the newest checkpoint in nick/model_parameters/PPO/
• Renders the agent for a few episodes (render_mode="human")
"""

from pathlib import Path
import multiprocessing as mp
import gymnasium as gym
from stable_baselines3 import PPO


ENV_ID        = "CartPole-v1"                  # keep in sync with training
CHECKPOINT_DIR = Path(f"nick/model_parameters/PPO_{ENV_ID}")
DEVICE        = "cuda"                          # use "cuda" if you have a GPU
EPISODES      = 5
MODEL         = PPO 

def main() -> None:
    # ------------------------------------------------------------------
    # 1. Locate the latest checkpoint saved by the training script
    # ------------------------------------------------------------------
    model_path = f"nick/model_parameters/PPO_CartPole-v1/ppo_{ENV_ID}.zip"
    print(model_path)
    print(f"Loading checkpoint → {model_path}")

    # ------------------------------------------------------------------
    # 2. Create the *same* environment you trained on
    # ------------------------------------------------------------------
    env = gym.make(ENV_ID, render_mode="human")    # continuous on-screen render:contentReference[oaicite:0]{index=0}

    # ------------------------------------------------------------------
    # 3. Load the model — don’t instantiate PPO first! :contentReference[oaicite:1]{index=1}
    # ------------------------------------------------------------------
    model = MODEL.load(model_path, env=env, device=DEVICE)
    print("✓ model loaded")

    # ------------------------------------------------------------------
    # 4. Roll a few episodes
    # ------------------------------------------------------------------
    for ep in range(1, EPISODES + 1):
        obs, info = env.reset(seed=ep)            # deterministic resets
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)  # new API :contentReference[oaicite:2]{index=2}
            # With render_mode="human" you don’t need an explicit env.render()

        print(f"Episode {ep} finished ✔ terminated:{terminated}, truncated:{truncated}")

    env.close()


if __name__ == "__main__":
    # Explicit start-method keeps Python 3.12 happy on every OS :contentReference[oaicite:3]{index=3}
    mp.set_start_method("spawn", force=True)
    main()
