import numpy as np
import random

class QLearningAgent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.99, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros((state_size, action_size))
        self.alpha = alpha          
        self.gamma = gamma          
        self.epsilon = epsilon      
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_size - 1)  
        else:
            return np.argmax(self.q_table[state])           

    def learn(self, state, action, reward, next_state, done):
        best_next_action = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next_action * (not done)
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

        if done and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_q_table(self):
        return self.q_table


