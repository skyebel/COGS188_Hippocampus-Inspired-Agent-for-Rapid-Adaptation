"""Exact (no-simulation) policy evaluation on a maze, for the evaluation metrics.

The mazes are small finite MDPs, so the quantities in the proposal's "Evaluation Metrics" section
can be computed exactly instead of estimated from noisy rollouts:

    J_e      expected return from the start state of the agent's GREEDY policy   (-> rho_e, recovery, AUC)
    V_beh    expected return of the agent's actual epsilon-greedy policy         (-> regret)
    J_star   expected return of the optimal policy in the current maze          (reference point 1)
    J_rand   expected return of the uniformly random policy                      (reference point 0)

Conventions (state them in the write-up):
  * returns are DISCOUNTED with the task discount `env.gamma` (0.99 at full scale), not the agent's gamma;
  * the 1000-step episode cap is ignored (infinite horizon) - it almost never binds for sensible policies;
  * the model describes the maze in the env's CURRENT phase, so build a fresh `ExactModel(env)` after
    `env.set_phase("post")`.

    model = ExactModel(env)
    ref   = model.reference()                          # J_star, J_rand, V_star, Q_star, optimal_policy, optimal_path
    ev    = model.evaluate_agent(agent)                # J_greedy, J_behavior
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from agents.replay import epsilon_greedy_probs


class ExactModel:
    def __init__(self, env):
        self.env = env
        S, A, slip = env.n_states, env.n_actions, env.slip
        self.S, self.A, self.gamma = S, A, env.gamma
        # P[(s,a), s'] : transition probability including the slip (executed action = a w.p. 1-slip,
        # each of the other 3 actions w.p. slip/3 - the same rule as MazeEnv.step)
        rows = np.arange(S * A)
        s_of, a_of = rows // A, rows % A
        r_idx, c_idx, val = [], [], []
        for a_exec in range(A):
            p = np.where(a_of == a_exec, 1.0 - slip, slip / (A - 1))
            r_idx.append(rows); c_idx.append(env._nxt[s_of, a_exec]); val.append(p)
        self.P = sp.csr_matrix((np.concatenate(val), (np.concatenate(r_idx), np.concatenate(c_idx))),
                               shape=(S * A, S))
        self.terminal = env.terminal.copy()
        # expected immediate reward of (s,a); terminal states are absorbing with value 0
        self.R = (self.P @ env.rew_enter).reshape(S, A)
        self.R[self.terminal] = 0.0
        self._cont = sp.diags((~self.terminal).astype(float))      # zeroes the rows of terminal states
        self._ref = None

    # ---- policy evaluation ------------------------------------------------------
    def value(self, pi):
        """V^pi (|S|,) for a stochastic policy pi of shape (|S|, |A|). Solves (I - gamma P_pi) V = R_pi exactly."""
        S, A = self.S, self.A
        Pi = sp.csr_matrix((np.asarray(pi).ravel(), (np.repeat(np.arange(S), A), np.arange(S * A))),
                           shape=(S, S * A))
        P_pi = self._cont @ (Pi @ self.P)
        R_pi = (np.asarray(pi) * self.R).sum(axis=1)
        return spla.spsolve(sp.identity(S, format="csc") - self.gamma * P_pi.tocsc(), R_pi)

    def q_from_v(self, V):
        return self.R + self.gamma * (self.P @ V).reshape(self.S, self.A) * (~self.terminal)[:, None]

    def optimal(self, max_iter=200):
        """(V*, Q*, greedy policy) by policy iteration (exact, converges in a few dozen solves)."""
        pol = np.zeros(self.S, dtype=int)
        for _ in range(max_iter):
            V = self.value(np.eye(self.A)[pol])
            Q = self.q_from_v(V)
            # keep the current action unless another is strictly better (avoids cycling on ties)
            better = Q.max(axis=1) > Q[np.arange(self.S), pol] + 1e-10
            if not better.any():
                break
            pol = np.where(better, Q.argmax(axis=1), pol)
        return V, Q, pol

    def reference(self):
        """J_star, J_rand and the optimal policy/path of the current maze (cached)."""
        if self._ref is None:
            s0 = self.env.start_state
            V, Q, pol = self.optimal()
            V_rand = self.value(np.full((self.S, self.A), 1.0 / self.A))
            self._ref = dict(J_star=float(V[s0]), J_rand=float(V_rand[s0]), V_star=V, Q_star=Q,
                             optimal_policy=pol, optimal_path=self.path(pol))
        return self._ref

    def path(self, policy, max_len=400):
        """Most-likely (no-slip) state sequence from the start under a deterministic policy; stops at a
        goal, when stuck, or after max_len. For the 'replay content' metric (is replay on the task path?)."""
        s = self.env.start_state
        out = [s]
        for _ in range(max_len):
            if self.terminal[s]:
                break
            s2 = int(self.env._nxt[s, int(policy[s])])
            if s2 == s:
                break
            s = s2
            out.append(s)
        return out

    # ---- agents ---------------------------------------------------------------------
    def evaluate_agent(self, agent):
        """Exact start-state returns of the agent's current policy.
        J_greedy   : always take the currently best action (ties split evenly)  -> J_e in the proposal
        J_behavior : the agent's actual epsilon-greedy policy                   -> V^{pi_beh}(s0) for regret"""
        Q, s0 = agent.q_table(), self.env.start_state
        return dict(J_greedy=float(self.value(epsilon_greedy_probs(Q, 0.0))[s0]),
                    J_behavior=float(self.value(epsilon_greedy_probs(Q, agent.epsilon))[s0]))
