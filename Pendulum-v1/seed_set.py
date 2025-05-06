import random
import numpy as np
import torch

class SeedSetter:
    def __init__(self, seed: int):
        self.seed = seed
        self.set_global_seed()

    def set_global_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    def apply_to_env(self, env):
        if hasattr(env, 'reset'):
            env.reset(seed=self.seed)
        if hasattr(env, 'action_space') and hasattr(env.action_space, 'seed'):
            env.action_space.seed(self.seed)
        if hasattr(env, 'observation_space') and hasattr(env.observation_space, 'seed'):
            env.observation_space.seed(self.seed)
