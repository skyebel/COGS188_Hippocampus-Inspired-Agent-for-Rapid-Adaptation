"""Model-based planner: value iteration on a learned model (upper-bound
reference for adaptation speed, at high compute cost)."""
import numpy as np
from .base import BaseAgent
from .model import LearnedModel


class ModelBasedPlanner(BaseAgent):
    name = "model_based_planner"

    def __init__(self, n_states, n_actions, forget=0.95, plan_every=10,
                 max_sweeps=500, tol=1e-4, **kwargs):
        """
        forget     : recency weighting of the learned model (see LearnedModel)
        plan_every : run value iteration every this many real steps
                     (1 = after every step; also always runs in end_episode()).
        max_sweeps : cap on VI sweeps per planning call (warm-started)
        """
        super().__init__(n_states, n_actions, **kwargs)
        self.model = LearnedModel(n_states, n_actions, forget)
        self.plan_every = int(plan_every)
        self.max_sweeps = int(max_sweeps)
        self.tol = float(tol)
        self.Q = np.zeros((n_states, n_actions))

    def q_values(self, state):
        return self.Q[state]

    def q_table(self):
        return self.Q

    def update(self, state, action, reward, next_state, done):
        self.n_real_steps += 1
        self.model.update(state, action, reward, next_state, done)
        if self.n_real_steps % self.plan_every == 0:
            self.plan()

    def end_episode(self):
        self.plan()

    def plan(self):
        """Warm-started value iteration on the learned model.
        n_backups grows by (#visited state-action pairs) per sweep."""
        m = self.model
        visited = m.visited
        n_vis = int(visited.sum())
        if n_vis == 0:
            return
        P = m.sparse_transitions()
        r_hat = m.reward_estimate()
        S, A = self.n_states, self.n_actions
        for _ in range(self.max_sweeps):
            V = self.Q.max(axis=1)
            V[m.terminal] = 0.0
            Q_new = r_hat + self.gamma * (P @ V).reshape(S, A)
            Q_new = np.where(visited, Q_new, self.Q)
            delta = np.abs(Q_new - self.Q).max()
            self.Q = Q_new
            self.n_backups += n_vis
            if delta < self.tol:
                break
