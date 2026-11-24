"""Multi-room maze layouts for the three Rat-Brained tasks.

All mazes are a 3x3 grid of square rooms separated by one-cell-thick walls with
a few one-cell doorways ("full" scale: 9x9 rooms -> 29x29 grid, ~740 open
cells, inside the 25x25-30x30 / 600-900 state range in the proposal).  A
"small" scale (3x3 rooms -> 11x11 grid, ~85 states) uses the SAME topology and
exists only for fast unit tests / pipeline demos.

Each task has a fixed layout (one per task, as in the proposal) with a
*pre-change* and a *post-change* version:

reward_revaluation      two goals, same layout; reward values are swapped
transition_revaluation  the high-reward goal's doorway is MOVED to the opposite
                        wall of its room (rewards unchanged); the preferred
                        goal flips because the long route to it changes
tolman_detour           one goal; a doorway on the usual route is BLOCKED
tolman_shortcut         (variant) the reverse: a new doorway OPENS

Design checks (policy flip, connectivity) live in ``verify_spec``.
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

Cell = Tuple[int, int]
MOVES = ((-1, 0), (1, 0), (0, 1), (0, -1))          # N, S, E, W
ACTION_NAMES = ("N", "S", "E", "W")


@dataclass(eq=False)
class MazeSpec:
    name: str
    task: str
    description: str
    walls_pre: np.ndarray                    # bool (H, W), True = wall
    walls_post: np.ndarray
    start: Cell
    goals_pre: Tuple[Tuple[Cell, float], ...]
    goals_post: Tuple[Tuple[Cell, float], ...]
    gamma: float = 0.99                      # task (evaluation) discount
    scale: str = "full"
    changed_cells: Tuple[Tuple[Cell, str], ...] = ()   # (cell, 'opened'|'closed') for plots

    @property
    def shape(self):
        return self.walls_pre.shape

    def open_cells_union(self):
        """Cells open in at least one phase (this defines the state space, so
        state indices are identical before and after the change)."""
        return ~(self.walls_pre & self.walls_post)


# ---------------------------------------------------------------------------
class RoomBuilder:
    """3x3 grid of ROOM x ROOM rooms; wall lines between rooms."""

    def __init__(self, room: int, n_rooms: int = 3):
        self.R, self.n = room, n_rooms
        self.size = n_rooms * room + (n_rooms - 1)
        self.step = room + 1

    def walls(self):
        w = np.zeros((self.size, self.size), bool)
        for i in range(1, self.n):
            w[i * self.step - 1, :] = True
            w[:, i * self.step - 1] = True
        return w

    def off(self, sym):
        """Symbolic offset inside a room: z=first cell, n=second ("near" the
        corner, so the start is not in a corner), m=middle, f=last cell."""
        R = self.R
        return {"z": 0, "n": min(1, R - 1), "m": R // 2, "f": R - 1}[sym]

    def pos(self, room, dr, dc):
        return (room[0] * self.step + self.off(dr), room[1] * self.step + self.off(dc))

    def door(self, a, b, k):
        """Cell in the wall between adjacent rooms a and b, at offset k (symbol)."""
        (r1, c1), (r2, c2) = a, b
        if r1 == r2:
            return (r1 * self.step + self.off(k), min(c1, c2) * self.step + self.R)
        return (min(r1, r2) * self.step + self.R, c1 * self.step + self.off(k))

    def open_doors(self, w, doors):
        w = w.copy()
        for a, b, k in doors:
            w[self.door(a, b, k)] = False
        return w

    def set_door(self, w, a, b, k, is_open):
        w = w.copy()
        w[self.door(a, b, k)] = not is_open
        return w


def _bfs(walls, start):
    H, W = walls.shape
    d = {start: 0}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in MOVES:
            n = (r + dr, c + dc)
            if 0 <= n[0] < H and 0 <= n[1] < W and not walls[n] and n not in d:
                d[n] = d[(r, c)] + 1
                q.append(n)
    return d


# ---------------------------------------------------------------------------
# door sets shared by the tasks ("f" = far end of the wall, "z" = near end)
_RR_DOORS = [((0, 0), (0, 1), "f"), ((0, 1), (0, 2), "z"), ((0, 2), (1, 2), "f"),     # top branch
             ((0, 0), (1, 0), "f"), ((1, 0), (2, 0), "z"), ((2, 0), (2, 1), "f"),     # left branch
             ((1, 2), (2, 2), "z"), ((2, 1), (2, 2), "f"),                            # loop closure
             ((1, 0), (1, 1), "f")]                                                   # centre alcove

_TR_PRE_DOORS = [((0, 0), (0, 1), "f"), ((0, 1), (0, 2), "f"), ((0, 2), (1, 2), "f"),  # top route -> hi goal
                 ((0, 0), (1, 0), "f"), ((1, 0), (2, 0), "z"), ((2, 0), (2, 1), "f"),  # left route -> lo goal
                 ((2, 1), (2, 2), "f"), ((1, 0), (1, 1), "f")]


def _spec_scale(scale):
    return {"full": RoomBuilder(9), "small": RoomBuilder(3)}[scale]


def reward_revaluation(scale="full", r_hi=1.0, r_lo=0.5):
    b = _spec_scale(scale)
    w = b.open_doors(b.walls(), _RR_DOORS)
    start = b.pos((0, 0), "n", "n")
    gA, gB = b.pos((1, 2), "f", "z"), b.pos((2, 1), "z", "f")      # equidistant by symmetry
    return MazeSpec(
        name="reward_revaluation", task="reward_revaluation",
        description="Two equidistant goals (A top-right branch, B bottom-left branch). After training the "
                    "reward values are swapped; the layout does not change.",
        walls_pre=w, walls_post=w.copy(), start=start,
        goals_pre=((gA, r_hi), (gB, r_lo)), goals_post=((gA, r_lo), (gB, r_hi)),
        gamma=_gamma(scale), scale=scale)


def transition_revaluation(scale="full", r_hi=1.0, r_lo=0.7):
    b = _spec_scale(scale)
    pre = b.open_doors(b.walls(), _TR_PRE_DOORS)
    post = b.set_door(pre, (0, 2), (1, 2), "f", False)              # close old doorway ...
    post = b.set_door(post, (1, 2), (2, 2), "f", True)              # ... doorway moved to the opposite wall
    hi, lo = b.pos((1, 2), "z", "z"), b.pos((2, 0), "z", "n")
    return MazeSpec(
        name="transition_revaluation", task="transition_revaluation",
        description="High-reward goal is reached through the top route. After training its doorway is moved "
                    "to the opposite wall of its room, so the old corridor dead-ends and the cached route is "
                    "stale. Rewards are unchanged; the nearer low-reward goal becomes the better choice.",
        walls_pre=pre, walls_post=post, start=b.pos((0, 0), "n", "n"),
        goals_pre=((hi, r_hi), (lo, r_lo)), goals_post=((hi, r_hi), (lo, r_lo)),
        gamma=_gamma(scale), scale=scale,
        changed_cells=((b.door((0, 2), (1, 2), "f"), "closed"), (b.door((1, 2), (2, 2), "f"), "opened")))


def tolman_detour(scale="full", variant="block", reward=1.0):
    """One goal. variant='block': a doorway on the usual route closes (detour).
    variant='shortcut': a new doorway opens (the reverse change)."""
    b = _spec_scale(scale)
    doors = _TR_PRE_DOORS + [((1, 2), (2, 2), "f")]                  # long alternative route exists
    open_all = b.open_doors(b.walls(), doors)
    blocked = b.set_door(open_all, (0, 1), (0, 2), "f", False)       # top route cut
    hi = b.pos((1, 2), "z", "z")
    d = b.door((0, 1), (0, 2), "f")
    if variant == "block":
        pre, post, name, ch = open_all, blocked, "tolman_detour", "closed"
        desc = "Single goal. A doorway on the learned short route is blocked after training; a ~2x longer route remains."
    else:
        pre, post, name, ch = blocked, open_all, "tolman_shortcut", "opened"
        desc = "Single goal. After training a new doorway opens, creating a ~2x shorter route (latent-learning test)."
    return MazeSpec(name=name, task="tolman_detour", description=desc,
                    walls_pre=pre, walls_post=post, start=b.pos((0, 0), "n", "n"),
                    goals_pre=((hi, reward),), goals_post=((hi, reward),),
                    gamma=_gamma(scale), scale=scale, changed_cells=((d, ch),))


def _gamma(scale):
    return 0.99 if scale == "full" else 0.95


def tiny_open(n=5, gamma=0.9):
    """n x n open grid, start top-left, goal bottom-right: the small 5x5 maze of the proposal's
    validation checks (slip rate ~10%, planner optimal, learned SR == closed-form SR)."""
    w = np.zeros((n, n), bool)
    goal = ((n - 1, n - 1), 1.0)
    return MazeSpec(name="tiny_open", task="tiny", description=f"{n}x{n} open grid for validation checks.",
                    walls_pre=w, walls_post=w.copy(), start=(0, 0), goals_pre=(goal,), goals_post=(goal,),
                    gamma=gamma, scale="tiny")


TASKS = ("reward_revaluation", "transition_revaluation", "tolman_detour")


def get_spec(name: str, scale: str = "full") -> MazeSpec:
    if name in ("tiny", "tiny_open"):
        return tiny_open()
    if name == "reward_revaluation":
        return reward_revaluation(scale)
    if name == "transition_revaluation":
        return transition_revaluation(scale)
    if name in ("tolman_detour", "tolman_detour_block"):
        return tolman_detour(scale, "block")
    if name == "tolman_shortcut":
        return tolman_detour(scale, "shortcut")
    raise KeyError(name)


def verify_spec(spec: MazeSpec) -> Dict[str, object]:
    """Structural sanity checks + shortest-path lengths; raises on a bad design."""
    out = {"n_open_union": int(spec.open_cells_union().sum()), "shape": spec.shape}
    for phase, walls, goals in (("pre", spec.walls_pre, spec.goals_pre), ("post", spec.walls_post, spec.goals_post)):
        assert not walls[spec.start], f"start is a wall ({phase})"
        d = _bfs(walls, spec.start)
        out[f"n_open_{phase}"] = int((~walls).sum())
        out[f"reachable_{phase}"] = len(d)
        out[f"dist_{phase}"] = {g: d.get(g) for g, _ in goals}
        for g, _ in goals:
            assert not walls[g], f"goal is a wall ({phase})"
            assert g in d, f"goal {g} unreachable ({phase})"
    # preferred goal (max discounted value on the shortest path) before/after
    def best(goals, dist):
        return max(goals, key=lambda gr: gr[1] * spec.gamma ** dist[gr[0]])[0]
    out["preferred_pre"] = best(spec.goals_pre, out["dist_pre"])
    out["preferred_post"] = best(spec.goals_post, out["dist_post"])
    if spec.task in ("reward_revaluation", "transition_revaluation"):
        assert out["preferred_pre"] != out["preferred_post"], f"{spec.name}: preferred goal does not flip"
    return out
