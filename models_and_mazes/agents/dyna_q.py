"""Dyna-Q with uniform replay: replay baseline (no SR, no prioritization)."""
import numpy as np
from .q_learning import QLearningAgent
from .replay import ExperienceStore


class DynaQAgent(QLearningAgent):
    name = "dyna_q"

    def __init__(self, n_states, n_actions, replay_budget=20, store_by_count=True,
                 store_forget=0.95, **kwargs):
        super().__init__(n_states, n_actions, **kwargs)
        self.replay_budget = int(replay_budget)   # k backups per real step
        self.store_by_count = store_by_count
        self.store = ExperienceStore(forget=store_forget)

    def _replay(self, origin, trigger):
        self._start_replay(trigger)
        for i in self.store.sample(self.rng, self.replay_budget, self.store_by_count):
            s, a, r, s2, d = self.store.get(i)
            self._backup(s, a, r, s2, d)
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
