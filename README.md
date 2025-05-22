# Intelligent-Control
# Intrinsic Motivation in Continuous Control Environments

This repository contains a comprehensive implementation and evaluation of various RL algorithms on two benchmark environments: **Pendulum-v1** and **Swimmer-v4**. The goal is to assess exploration strategies and robustness using both classical and deep RL approaches under noisy settings and intrinsic motivation.

---

## Environments
- **Pendulum-v1** (classic control, continuous action)
- **Swimmer-v4** (MuJoCo-based, high-dimensional control)

---

## Algorithms Compared

### On Pendulum-v1:
| Algorithm                    | Description |
|-----------------------------|-------------|
| `Q_learning_pendulum.py`    | Discretized Q-learning with noisy observations |
| `entropy_maximization_pendulum.py` | Q-learning with entropy bonus for improved exploration |
| `noise_based_explore_pendulum.py` | DDPG with noise-adaptive perturbation |
| `PPO_pendulum.py`           | PPO baseline with noisy observations |
| `PPO_intrinsic_pendulum.py` | PPO with intrinsic RND-based shaping rewards |

### On Swimmer-v4:
| Algorithm                    | Description |
|-----------------------------|-------------|
| `DDPG_noise_swimmer.py`     | DDPG with adaptive noise perturbation |
| `PPO_intrinsic_swimmer.py`  | PPO with RND intrinsic motivation and shaping bonuses |

---

## Evaluation Scripts

These scripts load trained models and run evaluation episodes with video recording:
- `test_case_pendulum.py` — Evaluates PPO + RND on Pendulum and records to `/videos/`
- `test_case_swimmer.py` — Evaluates PPO + RND on Swimmer and records to `/videos/`

---

## Shared Plot Utilities

Generate reward comparison plots:
- `shared_plots_pendulum.py` — Pendulum results across Q-learning, PPO, DDPG variants
- `shared_plots_swimmer.py` — Swimmer comparison between PPO + RND and DDPG

---

## Reproducibility

All experiments use a shared seed via `seed_set.py` to ensure deterministic behavior.

```python
seeder = SeedSetter(seed=42)
seeder.apply_to_env(env)
```

---

## Running the Project

### Install dependencies

```bash
pip install torch numpy matplotlib stable-baselines3 gymnasium moviepy imageio[ffmpeg]
```

### Train models

```bash
python Q_learning_pendulum.py
python entropy_maximization_pendulum.py
python noise_based_explore_pendulum.py
python PPO_pendulum.py
python PPO_intrinsic_pendulum.py
python DDPG_noise_swimmer.py
python PPO_intrinsic_swimmer.py
```

### Plot comparisons

```bash
python shared_plots_pendulum.py
python shared_plots_swimmer.py
```

### Evaluate with video recording

```bash
python test_case_pendulum.py
python test_case_swimmer.py
```

---

## Output Files

- **Logs:** in `/logs/` as `.npy` arrays of reward values
- **Models:** saved as `.zip` files by Stable Baselines3
- **Videos:** stored under `/videos/` as `.mp4` episodes for demonstration

---

## Conclusion

This study demonstrates:
- How intrinsic motivation (RND) improves exploration in PPO
- How entropy and noise-based perturbations enhance classic methods
- A consistent framework for evaluating diverse RL strategies under noise

All components are modular, extensible, and benchmark-ready for future experiments.
