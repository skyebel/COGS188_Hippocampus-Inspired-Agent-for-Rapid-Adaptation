"""Prioritized replay on a Q-table (original Mattar & Daw 2018 agent).
Ablation that removes the SR: replay is ordered by EVB = need x gain, with
need computed from a learned transition model instead of an SR."""
import numpy as np
from .q_learning import QLearningAgent
from .model import LearnedModel
from .replay import ExperienceStore, epsilon_greedy_probs, prioritized_replay


class PrioritizedQAgent(QLearningAgent):
    name = "prioritized_q"

    def __init__(self, n_states, n_actions, replay_budget=20, model_forget=0.95,
                 need_horizon=100, evb_floor=1e-10, evb_threshold=0.0,
                 max_seq_len=1, **kwargs):
        """max_seq_len > 1 enables n-step (sequence-extending) backups (H3 test)."""
        super().__init__(n_states, n_actions, **kwargs)
        self.replay_budget = int(replay_budget)
        self.need_horizon = int(need_horizon)
        self.evb_floor = evb_floor
        self.evb_threshold = evb_threshold
        self.max_seq_len = int(max_seq_len)
        self.store = ExperienceStore(forget=model_forget)   # same recency as the model
        self.model = LearnedModel(n_states, n_actions, model_forget)
        self._origin = 0

    # ---- callbacks for replay.prioritized_replay ----------------------------
    def q_rows(self, states):
        return self.Q[states]

    def backup_values(self, idx):
        st = self.store
        s, a, r, s2, d = (st.states[idx], st.actions[idx], st.rewards[idx],
                          st.next_states[idx], st.dones[idx])
        target = r + self.gamma * (~d) * self.Q[s2].max(axis=1)
        return self.Q[s, a] + self.alpha * (target - self.Q[s, a])

    def apply_backup(self, i, evb, need, gain):
        s, a, r, s2, d = self.store.get(i)
        self._backup(s, a, r, s2, d)
        self._log_backup(s, a, s2, origin=self._origin, evb=evb, need=need, gain=gain)

    def _chain_target(self, chain):
        st, G, disc = self.store, 0.0, 1.0
        for e in chain:
            G += disc * st.rewards[e]
            disc *= self.gamma
        last = chain[-1]
        if not st.dones[last]:
            G += disc * self.Q[st.next_states[last]].max()
        return G

    def chain_backup_value(self, chain):
        s0, a0 = int(self.store.states[chain[0]]), int(self.store.actions[chain[0]])
        return self.Q[s0, a0] + self.alpha * (self._chain_target(chain) - self.Q[s0, a0])

    def apply_chain(self, chain, evb, need, gain):
        st = self.store
        s0, a0 = int(st.states[chain[0]]), int(st.actions[chain[0]])
        self.Q[s0, a0] += self.alpha * (self._chain_target(chain) - self.Q[s0, a0])
        j = chain[-1]
        self._log_backup(int(st.states[j]), int(st.actions[j]), int(st.next_states[j]),
                         origin=self._origin, evb=evb, need=need, gain=gain, seq_len=len(chain))

    # ---- replay batch -----------------------------------------------------------
    def _replay(self, origin, trigger):
        self._start_replay(trigger)
        self._origin = origin
        if self.replay_budget > 0 and len(self.store):
            pi = epsilon_greedy_probs(self.Q, self.epsilon)
            need = self.model.need(origin, pi, self.gamma, self.need_horizon)
            prioritized_replay(self, self.store, need, self.replay_budget, self.epsilon,
                               self.evb_floor, self.evb_threshold, self.max_seq_len)

    def begin_episode(self, start_state=None):
        super().begin_episode(start_state)
        if self.replay_at_start and self.replay_budget > 0 and len(self.store) and self.start_state is not None:
            self._replay(self.start_state, (0.0, False, np.nan, True))

    # ---- main step ----------------------------------------------------------
    def update(self, state, action, reward, next_state, done):
        delta = super().update(state, action, reward, next_state, done)
        self.model.update(state, action, reward, next_state, done)
        self.store.add(state, action, reward, next_state, done)
        self._replay(self._origin_after(next_state, done), (reward, done, delta, False))
        return delta
