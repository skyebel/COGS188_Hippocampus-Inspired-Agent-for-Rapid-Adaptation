"""SR-Dyna (uniform replay): SR agent + offline replay of stored transitions,
sampled uniformly. Ablation that isolates the effect of prioritization."""
import numpy as np
from .sr_td import SRTDAgent
from .replay import ExperienceStore


class SRDynaAgent(SRTDAgent):
    name = "sr_dyna"

    def __init__(self, n_states, n_actions, replay_budget=20, store_by_count=True,
                 store_forget=0.95, **kwargs):
        super().__init__(n_states, n_actions, **kwargs)
        self.replay_budget = int(replay_budget)
        self.store_by_count = store_by_count
        self.store = ExperienceStore(forget=store_forget)

    def _sr_backup(self, s, a, s2, done):
        """Off-policy SR backup: bootstrap from a'* = argmax_a' H(s'a',.)·w."""
        sa = self.sa(s, a)
        if done:
            next_sa = None
        else:
            a_star = int(np.argmax(self.q_values(s2)))
            next_sa = self.sa(s2, a_star)
        self._h_update(sa, next_sa)

    def _replay(self, origin, trigger):
        """One batch of replay (overridden by SRPrioritizedAgent)."""
        self._start_replay(trigger)
        for i in self.store.sample(self.rng, self.replay_budget, self.store_by_count):
            s, a, _, s2, d = self.store.get(i)
            self._sr_backup(s, a, s2, d)
            self._log_backup(s, a, s2, origin=origin)

    def begin_episode(self, start_state=None):
        super().begin_episode(start_state)
        if self.replay_at_start and self.replay_budget > 0 and len(self.store) and self.start_state is not None:
            self._replay(self.start_state, (0.0, False, np.nan, True))

    def update(self, state, action, reward, next_state, done):
        delta = super().update(state, action, reward, next_state, done)
        self.store.add(state, action, reward, next_state, done)
        self._replay(self._origin_after(next_state, done), (reward, done, delta, False))
        return delta
