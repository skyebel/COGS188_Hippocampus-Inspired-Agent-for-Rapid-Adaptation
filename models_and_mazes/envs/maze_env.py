"""Gymnasium-compatible multi-room maze with stochastic ("slippery") moves.

State  : integer index of the agent's cell (one-hot for the SR). Indices are over
         the cells open in EITHER phase, so they are identical before/after the
         change; cells that are walls in the current phase are simply unreachable.
Action : 0..3 = N, S, E, W
Slip   : with prob. `slip` (default 0.10) the executed action is replaced by one of
         the OTHER three directions chosen uniformly (so the slip rate is exactly
         `slip`). A move into a wall leaves the agent in place.
Reward : given only on entering a goal cell (goal values may differ). Goals are
         terminal. Episodes are truncated at `max_steps` (default 1000).
Seeding: ``reset(seed=s)`` seeds the env RNG that drives the slips.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .layouts import MOVES, MazeSpec, get_spec


class MazeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, spec: MazeSpec, slip: float = 0.10, max_steps: int = 1000, phase: str = "pre"):
        super().__init__()
        self.maze, self.slip, self.max_steps = spec, float(slip), int(max_steps)
        union = spec.open_cells_union()
        self.cells = np.argwhere(union)                              # (S, 2) row-major
        self.n_states, self.n_actions = len(self.cells), 4
        self.state_of = -np.ones(spec.shape, np.int64)
        self.state_of[union] = np.arange(self.n_states)
        self.start_state = int(self.state_of[spec.start])
        self.observation_space = spaces.Discrete(self.n_states)
        self.action_space = spaces.Discrete(self.n_actions)
        self.gamma = spec.gamma
        self.phase = None
        self.set_phase(phase)
        self.s = self.start_state
        self.t = 0

    # ---- phases -------------------------------------------------------------
    def set_phase(self, phase: str):
        """'pre' (training layout/rewards) or 'post' (changed)."""
        assert phase in ("pre", "post")
        sp = self.maze
        walls = sp.walls_pre if phase == "pre" else sp.walls_post
        goals = sp.goals_pre if phase == "pre" else sp.goals_post
        S, A = self.n_states, self.n_actions
        nxt = np.empty((S, A), np.int64)
        H, W = walls.shape
        for s, (r, c) in enumerate(self.cells):
            for a, (dr, dc) in enumerate(MOVES):
                rr, cc = r + dr, c + dc
                ok = 0 <= rr < H and 0 <= cc < W and not walls[rr, cc] and not walls[r, c]
                nxt[s, a] = self.state_of[rr, cc] if ok else s
        self._nxt = nxt
        self.rew_enter = np.zeros(S)
        self.terminal = np.zeros(S, bool)
        self.goal_states = []
        for cell, val in goals:
            g = int(self.state_of[cell])
            self.rew_enter[g], self.terminal[g] = val, True
            self.goal_states.append(g)
        self.walls = walls
        self.phase = phase

    # ---- gymnasium API ----------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.s, self.t = self.start_state, 0
        return self.s, {"phase": self.phase}

    def step(self, action):
        a = int(action)
        slipped = False
        if self.np_random.random() < self.slip:
            a_exec = int(self.np_random.integers(self.n_actions - 1))
            a_exec = a_exec + 1 if a_exec >= a else a_exec          # uniform over the other 3 actions
            slipped = True
        else:
            a_exec = a
        s2 = int(self._nxt[self.s, a_exec])
        r = float(self.rew_enter[s2])
        terminated = bool(self.terminal[s2])
        self.s = s2
        self.t += 1
        truncated = (not terminated) and self.t >= self.max_steps
        return s2, r, terminated, truncated, {"slipped": slipped, "executed_action": a_exec}

    # ---- helpers for plots / analyses -----------------------------------------------
    def cell_of(self, s):
        return tuple(int(x) for x in self.cells[s])

    def to_grid(self, vec, fill=np.nan):
        """Scatter a per-state vector onto the (H, W) grid (walls/closed cells = fill)."""
        g = np.full(self.maze.shape, fill, float)
        g[self.cells[:, 0], self.cells[:, 1]] = vec
        g[self.walls] = fill
        return g

    def adjacency(self):
        """(S, S) bool matrix: True if two states are 4-neighbours and both open in the current phase.
        This is the weight matrix w_ij for Moran's I (SR shape / place-field metric)."""
        om = self.open_mask()
        A = np.zeros((self.n_states, self.n_states), bool)
        for s in np.flatnonzero(om):
            r, c = self.cells[s]
            for dr, dc in MOVES:
                rr, cc = r + dr, c + dc
                if 0 <= rr < self.maze.shape[0] and 0 <= cc < self.maze.shape[1]:
                    t = self.state_of[rr, cc]
                    if t >= 0 and om[t]:
                        A[s, t] = True
        return A

    def open_mask(self):
        """Per-state bool: cell is open in the current phase."""
        return ~self.walls[self.cells[:, 0], self.cells[:, 1]]


def make_env(task="reward_revaluation", scale="full", slip=0.10, max_steps=1000, phase="pre"):
    return MazeEnv(get_spec(task, scale), slip=slip, max_steps=max_steps, phase=phase)
