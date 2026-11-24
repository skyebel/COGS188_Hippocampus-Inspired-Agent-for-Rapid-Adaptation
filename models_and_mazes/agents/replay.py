"""Shared replay machinery: experience store, EVB (gain) computation, and
helpers for classifying replay direction.

Used by DynaQAgent, SRDynaAgent, PrioritizedQAgent and SRPrioritizedAgent so
that the replay-based agents differ ONLY in what they replay and how they pick
it (clean ablations for H4).
"""
import numpy as np

TIE_ATOL = 1e-8

# One row per replay backup. `replay_id` identifies the batch (block of k backups)
# it belongs to; `order` is its position inside that batch; `seq_len` is the length
# of the n-step chain it extended (1 for ordinary one-step backups); the trig_* fields
# describe the event that triggered the batch (a real step, or the start of an episode).
REPLAY_LOG_FIELDS = ("real_step", "replay_id", "order", "state", "action", "next_state",
                     "origin", "evb", "need", "gain", "seq_len",
                     "trig_reward", "trig_done", "trig_delta", "trig_start")
REPLAY_INT_FIELDS = ("real_step", "replay_id", "order", "state", "action", "next_state",
                     "origin", "seq_len")


def epsilon_greedy_probs(Q, epsilon):
    """Tie-aware epsilon-greedy action probabilities.

    Q: (..., A). Returns array of the same shape; greedy mass is split evenly
    among tied maximal actions.
    """
    Q = np.asarray(Q)
    A = Q.shape[-1]
    mx = Q.max(axis=-1, keepdims=True)
    tie = np.isclose(Q, mx, rtol=0.0, atol=TIE_ATOL)
    return epsilon / A + (1.0 - epsilon) * tie / tie.sum(axis=-1, keepdims=True)


def policy_gain(Q_rows, actions, q_new, epsilon):
    """Gain of a candidate backup (Mattar & Daw 2018).

    Gain = sum_a Q_new(s,a) * [pi_new(a|s) - pi_old(a|s)]
    where only Q(s, a_k) changes (to q_new) and pi is epsilon-greedy.

    Q_rows : (N, A) current Q-values of each candidate's state
    actions: (N,)   the action being backed up
    q_new  : (N,)   Q(s, a) after the backup
    """
    idx = np.arange(len(actions))
    Q_new = Q_rows.astype(np.float64, copy=True)
    Q_new[idx, actions] = q_new
    pi_old = epsilon_greedy_probs(Q_rows, epsilon)
    pi_new = epsilon_greedy_probs(Q_new, epsilon)
    return ((pi_new - pi_old) * Q_new).sum(axis=1)


class ExperienceStore:
    """Episodic memory of *unique* transitions (s, a, s', done).

    Each unique transition keeps its latest observed reward and a (possibly
    decayed) visit count. Storing unique transitions keeps prioritized replay
    tractable (candidate set <= |S|*|A|*branching).

    forget < 1 gives the store the same recency weighting as LearnedModel: every
    time (s, a) is observed, the counts of ALL outcomes recorded for that (s, a)
    are multiplied by `forget` before the observed one is incremented. Outcomes
    whose count falls below `prune_below` are deleted. Without this, after a
    transition change (rewired corridor, blocked path) the *old* (s, a, s')
    stays in memory forever and is replayed with its stale successor, so
    replay keeps re-teaching the pre-change world. forget=1.0 reproduces the
    old behaviour (never forgets).
    """

    _ARRS = ("_s", "_a", "_s2", "_r", "_done", "_count")

    def __init__(self, capacity=1024, forget=1.0, prune_below=0.05):
        self.forget = float(forget)
        self.prune_below = float(prune_below)
        self._keys = {}
        self._by_sa = {}
        self._cap = int(capacity)
        self.n = 0
        self._s = np.empty(self._cap, np.int32)
        self._a = np.empty(self._cap, np.int32)
        self._s2 = np.empty(self._cap, np.int32)
        self._r = np.empty(self._cap, np.float64)
        self._done = np.empty(self._cap, bool)
        self._count = np.empty(self._cap, np.float64)

    def __len__(self):
        return self.n

    def _grow(self):
        self._cap *= 2
        for name in self._ARRS:
            old = getattr(self, name)
            new = np.empty(self._cap, old.dtype)
            new[: self.n] = old[: self.n]
            setattr(self, name, new)

    def _compact(self, drop):
        keep = np.ones(self.n, bool)
        keep[list(drop)] = False
        m = int(keep.sum())
        for name in self._ARRS:
            arr = getattr(self, name)
            arr[:m] = arr[: self.n][keep]
        self.n = m
        self._keys, self._by_sa = {}, {}
        for j in range(m):
            s, a = int(self._s[j]), int(self._a[j])
            self._keys[(s, a, int(self._s2[j]), bool(self._done[j]))] = j
            self._by_sa.setdefault((s, a), []).append(j)

    def add(self, s, a, r, s2, done):
        """Insert / update a transition. Returns its (current) index."""
        key = (int(s), int(a), int(s2), bool(done))
        sa = key[:2]
        if self.forget < 1.0:
            for j in self._by_sa.get(sa, ()):
                self._count[j] *= self.forget
        i = self._keys.get(key)
        if i is None:
            if self.n == self._cap:
                self._grow()
            i = self.n
            self._keys[key] = i
            self._by_sa.setdefault(sa, []).append(i)
            self._s[i], self._a[i], self._s2[i] = s, a, s2
            self._done[i] = bool(done)
            self._count[i] = 0.0
            self.n += 1
        self._r[i] = r
        self._count[i] += 1.0
        if self.forget < 1.0:
            stale = [j for j in self._by_sa[sa] if j != i and self._count[j] < self.prune_below]
            if stale:
                self._compact(stale)
                i = self._keys[key]
        return i

    # views over the filled portion
    @property
    def states(self): return self._s[: self.n]
    @property
    def actions(self): return self._a[: self.n]
    @property
    def next_states(self): return self._s2[: self.n]
    @property
    def rewards(self): return self._r[: self.n]
    @property
    def dones(self): return self._done[: self.n]
    @property
    def counts(self): return self._count[: self.n]

    def get(self, i):
        return (int(self._s[i]), int(self._a[i]), float(self._r[i]),
                int(self._s2[i]), bool(self._done[i]))

    def sample(self, rng, k, by_count=True):
        """Uniform replay: k indices (with replacement)."""
        if self.n == 0:
            return np.empty(0, np.int64)
        if by_count:
            p = self._count[: self.n].astype(np.float64)
            p /= p.sum()
            return rng.choice(self.n, size=k, p=p)
        return rng.integers(0, self.n, size=k)


def _best_extension(agent, store, seq, need, epsilon, evb_floor):
    """Candidate n-step backup that extends the current replay chain `seq` by one
    more stored experience (Mattar & Daw 2018, 'sequence' backups).

    The chain e_1..e_n (e_{i+1}.s == e_i.s') updates Q(s_1, a_1) towards the n-step
    target  sum_i gamma^(i-1) r_i + gamma^n max_a Q(s'_n, a).  The experience that
    extends it is the stored experience leaving s'_n with the currently greedy
    action (most frequent outcome on ties).  Returns (j, evb, gain) or None."""
    S, A = store.states, store.actions
    end = int(store.next_states[seq[-1]])
    cand = np.flatnonzero(S == end)
    if len(cand) == 0:
        return None
    q_end = agent.q_rows(np.array([end]))[0]
    qa = q_end[A[cand]]
    cand = cand[np.isclose(qa, qa.max(), rtol=0.0, atol=TIE_ATOL)]
    j = int(cand[np.argmax(store.counts[cand])])
    chain = seq + [j]
    q_new = agent.chain_backup_value(chain)
    s0, a0 = int(S[seq[0]]), int(A[seq[0]])
    g = float(policy_gain(agent.q_rows(np.array([s0])), np.array([a0]), np.array([q_new]), epsilon)[0])
    return j, float(need[s0]) * max(g, evb_floor), g


def prioritized_replay(agent, store, need, budget, epsilon,
                       evb_floor=1e-10, evb_threshold=0.0, max_seq_len=1):
    """Greedy EVB-ordered replay (Mattar & Daw 2018), EVB = need x gain.

    The agent must provide:
      agent.q_rows(states)            -> (N, A) current Q for those states
      agent.backup_values(idx)        -> (N,) Q(s,a) *after* backing up store[idx]
      agent.apply_backup(i, evb, need, gain)  -> performs + logs the backup
    and, only if max_seq_len > 1 (n-step / sequence-extending backups, used to
    test forward replay, H3):
      agent.chain_backup_value(chain) -> Q(s_1,a_1) after the n-step backup of `chain`
      agent.apply_chain(chain, evb, need, gain)
    `need` is a length-|S| vector computed once for this replay batch.

    Gain is computed once for all candidates, then refreshed only for
    candidates whose gain can have changed after a backup at state s:
    those with state == s (their Q row changed) or next_state == s (their
    bootstrap target changed).

    Each iteration picks the best one-step backup, or - when max_seq_len > 1 and
    a chain exists - the chain extension if its EVB is strictly larger.
    """
    n = len(store)
    if n == 0 or budget <= 0:
        return 0
    S, A, S2 = store.states, store.actions, store.next_states

    def gains(idx):
        return policy_gain(agent.q_rows(S[idx]), A[idx],
                           agent.backup_values(idx), epsilon)

    gain = gains(np.arange(n))
    done_backups, seq = 0, []
    for _ in range(budget):
        evb = need[S] * np.maximum(gain, evb_floor)
        i = int(np.argmax(evb))
        best = float(evb[i])
        ext = None
        if max_seq_len > 1 and seq and len(seq) < max_seq_len and not store.dones[seq[-1]]:
            ext = _best_extension(agent, store, seq, need, epsilon, evb_floor)
        if ext is not None and ext[1] > best:
            j, evb_e, gain_e = ext
            seq = seq + [j]
            agent.apply_chain(seq, evb_e, float(need[S[seq[0]]]), gain_e)
        else:
            if best <= evb_threshold:
                break
            agent.apply_backup(i, best, float(need[S[i]]), float(gain[i]))
            seq = [i]
        done_backups += 1
        s_upd = S[seq[0]]
        aff = np.flatnonzero((S == s_upd) | (S2 == s_upd))
        gain[aff] = gains(aff)
    return done_backups


def classify_replay_direction(log):
    """Label each logged backup relative to the previous backup in the same
    replay batch: 'forward' (this backup's state == previous next_state),
    'reverse' (this backup's next_state == previous state), 'other', or
    'start' (first backup of a batch).

    `log` is the dict returned by agent.replay_log_arrays()."""
    batch = log["replay_id"] if "replay_id" in log else log["real_step"]
    s, s2 = log["state"], log["next_state"]
    out = np.full(len(batch), "start", dtype=object)
    for i in range(1, len(batch)):
        if batch[i] != batch[i - 1]:
            continue
        if s[i] == s2[i - 1]:
            out[i] = "forward"
        elif s2[i] == s[i - 1]:
            out[i] = "reverse"
        else:
            out[i] = "other"
    return out


def classify_replay_context(log):
    """H3 context label per backup, from the event that triggered its batch:
    'post_reward'  - the triggering real step delivered reward (replay right after
                     reward; Mattar & Daw predict mostly REVERSE here)
    'pre_move'     - batch run at the start of an episode, before the first move
                     (needs agent(replay_at_start=True); predicted mostly FORWARD)
    'pre_decision' - any other real step (replay between decisions).
    Cross-tabulate with classify_replay_direction()."""
    rew = np.asarray(log["trig_reward"]) != 0
    start = np.asarray(log["trig_start"]).astype(bool) if "trig_start" in log else np.zeros(len(rew), bool)
    return np.where(start, "pre_move", np.where(rew, "post_reward", "pre_decision"))
