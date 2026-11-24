"""Tabular Q-learning: model-free baseline (dopaminergic TD learning)."""
import numpy as np
from .base import BaseAgent


class QLearningAgent(BaseAgent):
    name = "q_learning"

    def __init__(self, n_states, n_actions, alpha=0.1, q_init=0.0, **kwargs):
        super().__init__(n_states, n_actions, **kwargs)
        self.alpha = float(alpha)
        self.Q = np.full((n_states, n_actions), q_init, dtype=np.float64)

    def q_values(self, state):
        return self.Q[state]

    def q_table(self):
        return self.Q

    def _backup(self, s, a, r, s2, done):
        target = r + (0.0 if done else self.gamma * self.Q[s2].max())
        delta = target - self.Q[s, a]
        self.Q[s, a] += self.alpha * delta
        return delta

    def update(self, state, action, reward, next_state, done):
        self.n_real_steps += 1
        return self._backup(state, action, reward, next_state, done)
