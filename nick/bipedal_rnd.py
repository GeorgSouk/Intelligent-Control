# tutorial2.py
import multiprocessing as mp
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from pathlib import Path

def main() -> None:
    ENV_ID, N_ENVS = "CartPole-v1", 2**3      
    ROLLOUT = 512                        # total steps per update
    N_STEPS = ROLLOUT // N_ENVS             # steps per env (must be ≥1)
    TOTAL_STEPS, LOOPS = 20_000, 20
    DEVICE = "cuda"                         # or "cpu"

    models_dir = Path(f"nick/model_parameters/PPO_{ENV_ID}")
    logs_dir   = Path(f"nick/model_logs/PPO_{ENV_ID}")
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True,  exist_ok=True)

    vec_env = make_vec_env(
        ENV_ID,
        n_envs      = N_ENVS,
        vec_env_cls = SubprocVecEnv,
        seed        = 0,
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps         = N_STEPS,
        tensorboard_log = str(logs_dir),
        device          = DEVICE,
        verbose         = 1,
    )

    for loop in range(1, LOOPS + 1):
        model.learn(
            total_timesteps     = TOTAL_STEPS,
            reset_num_timesteps = False,
            progress_bar        = True,
        )
        # if loop % 2 == 0:
        ckpt = models_dir / f"ppo_{ENV_ID}.zip"
        model.save(ckpt)
        print(f"[✓] saved → {ckpt}")
        print(f"loop{loop}")
       

    vec_env.close()

if __name__ == "__main__":
    # Choose the safest start method for your OS
    # mp.set_start_method("spawn", force=True)   # Windows/macOS
    mp.set_start_method("fork",  force=True) # Linux (faster, but be careful)
    main()
