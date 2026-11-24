"""Common agent interface.

Every agent works on a discrete (tabular) MDP with integer states/actions, so
any gymnasium.Env that exposes a discrete state index will plug in:

    agent = SomeAgent(n_states, n_actions, gamma=0.95, epsilon=0.1, seed=0)
    s, _ = env.reset(seed=0)
    agent.begin_episode(s)
    while True:
        a = agent.select_action(s)
        s2, r, terminated, truncated, _ = env.step(a)
        agent.update(s, a, r, s2, terminated)   # NOT `truncated`
        s = s2
        if terminated or truncated: break

IMPORTANT: pass `terminated` (true terminal) as `done`, not time-limit
truncation, otherwise value bootstrapping is wrong.
"""
from abc import ABC, abstractmethod
import numpy as np

from .replay import REPLAY_LOG_FIELDS, REPLAY_INT_FIELDS, epsilon_greedy_probs


class BaseAgent(ABC):
    name = "base"

    def __init__(self, n_states, n_actions, gamma=0.95, epsilon=0.1,
                 seed=None, log_replay=False, replay_at_start=False):
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)       # may be changed between phases
        self.rng = np.random.default_rng(seed)

        # compute accounting
        self.n_real_steps = 0               # environment steps seen
        self.n_backups = 0                  # offline / replay / planning backups

        # replay logging (for forward-vs-reverse analyses). Rows are buffered in a
        # python list and flushed into float32 chunks to keep memory small
        # (k=20 backups x 1e5 steps would otherwise be millions of tuples).
        # `log_replay` may be switched on/off at run time (e.g. only post-change).
        self.log_replay = bool(log_replay)
        self.replay_at_start = bool(replay_at_start)   # extra replay before the first move of each episode (H3 'pre_move')
        self._replay_buf = []
        self._replay_chunks = []
        self._replay_id = 0                 # one id per replay *batch* (a block of k backups)
        self._step_order = 0                # position of the backup inside its batch
        # (reward, done, TD error, at_episode_start) of the event that triggered the replay batch
        self._trigger = (0.0, False, np.nan, False)

        self.start_state = None
        self._pending = None                # queued next action (SARSA-style agents)

    # ---- required interface -------------------------------------------------
    @abstractmethod
    def q_values(self, state):
        """Q(state, .) as a length-|A| array."""

    @abstractmethod
    def update(self, state, action, reward, next_state, done):
        """Learn from one real transition (and run any offline replay)."""

    # ---- shared behaviour ---------------------------------------------------
    def q_table(self):
        return np.stack([self.q_values(s) for s in range(self.n_states)])

    def greedy_policy(self):
        """Deterministic greedy action per state, shape (|S|,)."""
        return self.q_table().argmax(axis=1)

    def policy_probs(self):
        """Epsilon-greedy action probabilities, shape (|S|, |A|)."""
        return epsilon_greedy_probs(self.q_table(), self.epsilon)

    def _argmax_random_tie(self, q):
        best = np.flatnonzero(np.isclose(q, q.max(), rtol=0.0, atol=1e-8))
        return int(best[0]) if len(best) == 1 else int(self.rng.choice(best))

    def greedy_action(self, state):
        return self._argmax_random_tie(self.q_values(state))

    def select_action(self, state, greedy=False):
        if not greedy and self._pending is not None and self._pending[0] == state:
            a = self._pending[1]
            self._pending = None
            return a
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return self.greedy_action(state)

    def begin_episode(self, start_state=None):
        """Call at env.reset(). The start state is used by replay agents to
        compute 'need' from the start location after reaching a goal."""
        self.start_state = start_state if start_state is not None else self.start_state
        self._pending = None

    def end_episode(self):
        """Optional hook at episode end (used by the planner)."""

    # ---- bookkeeping --------------------------------------------------------
    def _origin_after(self, next_state, done):
        """State from which 'need' is computed: after a goal the agent restarts at
        the start state (Mattar & Daw), otherwise it is at next_state."""
        return self.start_state if (done and self.start_state is not None) else next_state

    def _start_replay(self, trigger):
        """Call once at the beginning of every replay batch."""
        self._replay_id += 1
        self._step_order = 0
        self._trigger = trigger

    def _log_backup(self, s, a, s2, origin=-1, evb=np.nan, need=np.nan, gain=np.nan, seq_len=1):
        self.n_backups += 1
        if self.log_replay:
            self._replay_buf.append((self.n_real_steps, self._replay_id, self._step_order,
                                     s, a, s2, origin, evb, need, gain, seq_len, *self._trigger))
            self._step_order += 1
            if len(self._replay_buf) >= 50_000:
                self._flush_replay_buf()

    def _flush_replay_buf(self):
        if self._replay_buf:
            self._replay_chunks.append(np.asarray(self._replay_buf, dtype=np.float32))
            self._replay_buf = []

    def replay_log_arrays(self):
        """Replay log as a dict of numpy arrays keyed by REPLAY_LOG_FIELDS
        (wrap in pandas.DataFrame if desired)."""
        self._flush_replay_buf()
        if not self._replay_chunks:
            return {f: np.empty(0) for f in REPLAY_LOG_FIELDS}
        cols = np.concatenate(self._replay_chunks).T
        out = dict(zip(REPLAY_LOG_FIELDS, cols))
        for f in REPLAY_INT_FIELDS:
            out[f] = out[f].astype(np.int64)
        for f in ("trig_done", "trig_start"):
            out[f] = out[f].astype(bool)
        return out

    def reset_counters(self):
        """Zero compute counters/logs (e.g. at the start of the post-change phase)
        without touching learned values."""
        self.n_real_steps = 0
        self.n_backups = 0
        self._replay_buf, self._replay_chunks = [], []

    def __repr__(self):
        return (f"{self.__class__.__name__}(|S|={self.n_states}, |A|={self.n_actions}, "
                f"gamma={self.gamma}, epsilon={self.epsilon})")
