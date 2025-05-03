import numpy as np
import random

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
