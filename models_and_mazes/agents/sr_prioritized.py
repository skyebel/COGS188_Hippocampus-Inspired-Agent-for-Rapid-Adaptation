"""SR + prioritized replay (need x gain): the MAIN agent.

Predictive map (state-action SR, H) + utility-driven replay. Need is read
directly from the SR row of the agent's current state; gain is the policy
improvement from a one-step SR backup (Mattar & Daw 2018 on Russek et al. 2017).

For a candidate experience (s, a, s'), the post-backup value has a closed form
    Q_new(sa) = Q(sa) + alpha_sr * ( w(sa) + gamma * Q(s', a'*) - Q(sa) ),
so EVB for every stored experience is computed from the cached Q vector
without touching H. With max_seq_len > 1 the n-step (sequence-extending)
backup of Mattar & Daw is also available:
    H(s1a1,.) <- (1-alpha) H(s1a1,.) + alpha [ sum_i gamma^(i-1) 1_{s_i a_i} + gamma^n H(s'_n a'*,.) ]
"""
import numpy as np
from .sr_dyna import SRDynaAgent
from .replay import epsilon_greedy_probs, prioritized_replay


class SRPrioritizedAgent(SRDynaAgent):
    name = "sr_prioritized"

    def __init__(self, n_states, n_actions, replay_budget=20,
                 evb_floor=1e-10, evb_threshold=0.0, max_seq_len=1, **kwargs):
        super().__init__(n_states, n_actions, replay_budget=replay_budget, **kwargs)
        self.evb_floor = evb_floor
        self.evb_threshold = evb_threshold
        self.max_seq_len = int(max_seq_len)
        self._Qvec = None            # cached H @ w, valid during a replay phase
        self._origin = 0

    # ---- need from the SR ---------------------------------------------------
    def need_vector(self, origin):
        """Expected discounted visits to each state from `origin`, read off
        the SR rows of `origin`: sum_a pi(a|origin) H(origin a, .) summed over
        the action axis."""
        S, A = self.n_states, self.n_actions
        pi = epsilon_greedy_probs(self._Qvec.reshape(S, A)[origin], self.epsilon)
        occ = pi @ self.H[origin * A:(origin + 1) * A]            # (|S||A|,)
        return occ.reshape(S, A).sum(axis=1).astype(np.float64)

    # ---- callbacks for replay.prioritized_replay ----------------------------
    def q_rows(self, states):
        return self._Qvec.reshape(self.n_states, self.n_actions)[states]

    def backup_values(self, idx):
        st = self.store
        s, a, s2, d = st.states[idx], st.actions[idx], st.next_states[idx], st.dones[idx]
        Q2d = self._Qvec.reshape(self.n_states, self.n_actions)
        sa = s * self.n_actions + a
        q = self._Qvec[sa].astype(np.float64)
        target = self.w[sa] + self.gamma * (~d) * Q2d[s2].max(axis=1)
        return q + self.alpha_sr * (target - q)

    def apply_backup(self, i, evb, need, gain):
        s, a, _, s2, d = self.store.get(i)
        self._sr_backup(s, a, s2, d)
        sa = self.sa(s, a)
        self._Qvec[sa] = self.H[sa] @ self.w          # (alias of self._Q; _h_update already refreshed it)
        self._log_backup(s, a, s2, origin=self._origin, evb=evb, need=need, gain=gain)

    def chain_backup_value(self, chain):
        st, A = self.store, self.n_actions
        Q2d = self._Qvec.reshape(self.n_states, A)
        G, disc = 0.0, 1.0
        for e in chain:
            G += disc * self.w[int(st.states[e]) * A + int(st.actions[e])]
            disc *= self.gamma
        last = chain[-1]
        if not st.dones[last]:
            G += disc * Q2d[st.next_states[last]].max()
        sa0 = int(st.states[chain[0]]) * A + int(st.actions[chain[0]])
        q = float(self._Qvec[sa0])
        return q + self.alpha_sr * (G - q)

    def apply_chain(self, chain, evb, need, gain):
        st, A, al = self.store, self.n_actions, self.alpha_sr
        sa0 = int(st.states[chain[0]]) * A + int(st.actions[chain[0]])
        new = (1.0 - al) * self.H[sa0]
        disc = 1.0
        for e in chain:
            new[int(st.states[e]) * A + int(st.actions[e])] += al * disc
            disc *= self.gamma
        last = chain[-1]
        if not st.dones[last]:
            s_n = int(st.next_states[last])
            a_star = int(np.argmax(self.q_values(s_n)))
            new += (al * disc) * self.H[s_n * A + a_star]
        self.H[sa0] = new
        self._Qvec[sa0] = self.H[sa0] @ self.w
        j = chain[-1]
        self._log_backup(int(st.states[j]), int(st.actions[j]), int(st.next_states[j]),
                         origin=self._origin, evb=evb, need=need, gain=gain, seq_len=len(chain))

    # ---- replay batch (replaces SRDyna's uniform replay) --------------------------
    def _replay(self, origin, trigger):
        self._start_replay(trigger)
        self._origin = origin
        if self.replay_budget > 0 and len(self.store):
            self._Qvec = self._Q                         # exact cache of H @ w (kept in sync by _h_update)
            need = self.need_vector(origin)
            prioritized_replay(self, self.store, need, self.replay_budget, self.epsilon,
                               self.evb_floor, self.evb_threshold, self.max_seq_len)
            self._Qvec = None
