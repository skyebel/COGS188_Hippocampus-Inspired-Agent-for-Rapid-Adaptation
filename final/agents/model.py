"""Learned tabular world model (shared by the model-based planner and the
Q-table prioritized-replay ablation)."""
import numpy as np
from scipy import sparse


class LearnedModel:
    """Count-based transition/reward model with optional recency weighting.

    forget < 1 multiplies the old counts of a (s, a) pair by `forget` every
    time it is observed again, so the model can track changes in the maze
    (effective memory ~ 1/(1-forget) visits per (s, a)). forget=1.0 gives plain
    maximum-likelihood counts that never unlearn.
    """

    def __init__(self, n_states, n_actions, forget=0.95):
        self.S, self.A = n_states, n_actions
        self.forget = float(forget)
        self.N = np.zeros((n_states, n_actions, n_states), np.float32)
        self.W = np.zeros((n_states, n_actions), np.float64)       # total weight
        self.Rsum = np.zeros((n_states, n_actions), np.float64)
        self.terminal = np.zeros(n_states, bool)

    def update(self, s, a, r, s2, done):
        f = self.forget
        self.N[s, a] *= f
        self.N[s, a, s2] += 1.0
        self.W[s, a] = f * self.W[s, a] + 1.0
        self.Rsum[s, a] = f * self.Rsum[s, a] + r
        if done:
            self.terminal[s2] = True

    @property
    def visited(self):
        return self.W > 0

    def reward_estimate(self):
        return np.divide(self.Rsum, self.W, out=np.zeros_like(self.Rsum), where=self.W > 0)

    def sparse_transitions(self):
        """P as a sparse (|S|*|A|, |S|) matrix; unvisited rows are all-zero."""
        flat_N = self.N.reshape(self.S * self.A, self.S)
        w = self.W.reshape(-1, 1)
        P = np.divide(flat_N, w, out=np.zeros_like(flat_N), where=w > 0)
        return sparse.csr_matrix(P)

    def need(self, origin, pi, gamma, horizon=100, tol=1e-6):
        """Expected discounted visits to every state starting from `origin`
        under policy `pi` (|S|, |A|), using the learned model.
        This is the Mattar & Daw 'need' term when no SR is available."""
        visited = self.visited
        pi_v = pi * visited
        row = pi_v.sum(axis=1, keepdims=True)
        pi_v = np.divide(pi_v, row, out=np.zeros_like(pi_v), where=row > 0)
        wgt = np.divide(pi_v, self.W, out=np.zeros_like(pi_v), where=self.W > 0)
        T = np.einsum("sa,sat->st", wgt.astype(np.float32), self.N)
        TT = sparse.csr_matrix(T).T.tocsr()
        v = np.zeros(self.S)
        v[origin] = 1.0
        total = v.copy()
        for _ in range(horizon):
            v = gamma * (TT @ v)
            total += v
            if v.sum() < tol:
                break
        return total
