"""SR-TD: state-action successor representation H with SARSA reward weights.
Hippocampal predictive map (Dayan 1993; Russek et al. 2017). No replay.

H(sa, s'a') = expected discounted future occupancy of s'a' after taking a in s.
Q(sa) = sum_{s'a'} H(sa, s'a') w(s'a')

Row/column index of a state-action pair is  sa = s * n_actions + a.
"""
import numpy as np
from .base import BaseAgent
from .replay import epsilon_greedy_probs


class SRTDAgent(BaseAgent):
    name = "sr_td"

    def __init__(self, n_states, n_actions, alpha_sr=0.1, alpha_td=0.05,
                 h_init="identity", dtype=np.float32,
                 reward_mode="td", alpha_w=0.3, **kwargs):
        """
        alpha_sr : learning rate of the SR update
        alpha_td : learning rate of the reward-weight (w) SARSA update
                   (keep modest: the effective step scales with ||H(sa,.)||^2)
        h_init   : 'identity' (H = I, Stachenfeld-style) or 'zeros'
        reward_mode : 'td'     - w trained by SARSA TD error x H(sa,.) (Russek et al. 2017
                                 SR-TD); w is a value-shaped weight vector.
                      'direct' - w(sa) is the running mean of the reward received on
                                 taking a in s, so Q = H w literally "SR x reward
                                 vector" as in the proposal; a reward change is
                                 reflected as soon as w is re-estimated and an H
                                 change is reflected immediately in Q.
        alpha_w  : learning rate of the 'direct' reward estimate
        dtype    : float32 keeps H (|SA| x |SA|) at ~52 MB for 900 states x 4 actions
        """
        super().__init__(n_states, n_actions, **kwargs)
        self.alpha_sr = float(alpha_sr)
        self.alpha_td = float(alpha_td)
        n_sa = n_states * n_actions
        self.H = np.eye(n_sa, dtype=dtype) if h_init == "identity" else np.zeros((n_sa, n_sa), dtype)
        self.w = np.zeros(n_sa, dtype=dtype)
        assert reward_mode in ("td", "direct")
        self.reward_mode, self.alpha_w = reward_mode, float(alpha_w)
        # Cached Q = H @ w (flat, length |S||A|), kept exactly in sync: a row update of H refreshes
        # one entry; in 'direct' mode w changes in one entry per step so Q is patched with one
        # column of H; in 'td' mode w changes everywhere so Q is recomputed. A full refresh every
        # `refresh_every` updates removes float32 drift.
        self._Q = (self.H @ self.w).astype(dtype)
        self.refresh_every = 20000
        self._n_upd = 0

    # ---- indexing / values --------------------------------------------------
    def sa(self, s, a):
        return int(s) * self.n_actions + int(a)

    def q_values(self, state):
        lo = state * self.n_actions
        return self._Q[lo:lo + self.n_actions]

    def q_table(self):
        return self._Q.reshape(self.n_states, self.n_actions)

    def refresh_Q(self):
        self._Q = (self.H @ self.w).astype(self.H.dtype)

    # ---- SR primitives (reused by SR-Dyna / SR-prioritized) -----------------
    def _h_update(self, sa, next_sa):
        """H(sa,.) <- H(sa,.) + alpha [1_sa + gamma H(next_sa,.) - H(sa,.)]
        (next_sa=None for terminal transitions)."""
        a = self.alpha_sr
        new = (1.0 - a) * self.H[sa]
        if next_sa is not None:
            new += (a * self.gamma) * self.H[next_sa]
        new[sa] += a
        self.H[sa] = new
        self._Q[sa] = self.H[sa] @ self.w

    def update(self, state, action, reward, next_state, done):
        self.n_real_steps += 1
        sa = self.sa(state, action)
        self._pending = None
        next_sa = None
        if not done:
            a2 = self.select_action(next_state)       # SARSA: pick a' now,
            self._pending = (next_state, a2)          # and replay it next step
            next_sa = self.sa(next_state, a2)

        q_sa = float(self._Q[sa])
        q_next = 0.0 if next_sa is None else float(self._Q[next_sa])
        delta = reward + self.gamma * q_next - q_sa
        if self.reward_mode == "direct":
            dw = self.alpha_w * (reward - float(self.w[sa]))
            if dw != 0.0:
                self.w[sa] += dw
                self._Q += dw * self.H[:, sa]
        else:
            self.w += self.alpha_td * delta * self.H[sa]  # w(i) += a_td * delta * H(sa,i)
            self.refresh_Q()
        self._h_update(sa, next_sa)
        self._n_upd += 1
        if self._n_upd % self.refresh_every == 0:
            self.refresh_Q()
        return delta

    # ---- analysis helpers ---------------------------------------------------
    def state_sr(self, policy=None):
        """State-level SR matrix M(s, s') (|S| x |S|): expected discounted
        occupancy of s' from s under `policy` (|S|,|A|; default = current
        epsilon-greedy policy). Use for place-field / eigenvector analyses."""
        S, A = self.n_states, self.n_actions
        occ = self.H.reshape(S, A, S, A).sum(axis=-1)           # (S, A, S)
        pi = epsilon_greedy_probs(self.q_table(), self.epsilon) if policy is None else policy
        return np.einsum("sa,sat->st", pi, occ)

    def reward_weights(self):
        """w reshaped to (|S|, |A|)."""
        return self.w.reshape(self.n_states, self.n_actions)
