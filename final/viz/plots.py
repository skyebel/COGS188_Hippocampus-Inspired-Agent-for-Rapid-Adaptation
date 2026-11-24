"""Plot helpers for mazes, values/policies, SR fields and replay (matplotlib only)."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

from envs.layouts import ACTION_NAMES, MOVES

WALL, FLOOR, CLOSED = "#2b2b3a", "#f4f1ea", "#d96a6a"
OPEN_NEW = "#7fc8a9"


def _grid_axes(ax, shape):
    H, W = shape
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_maze(spec, phase="pre", ax=None, show_changes=True, title=None, label_goals=True):
    """Draw one phase of a MazeSpec: walls, start (S), goals (G + reward), changed doorways."""
    ax = ax or plt.subplots(figsize=(5, 5))[1]
    walls = spec.walls_pre if phase == "pre" else spec.walls_post
    goals = spec.goals_pre if phase == "pre" else spec.goals_post
    img = np.where(walls, 0, 1)
    ax.imshow(img, cmap=ListedColormap([WALL, FLOOR]), interpolation="nearest", vmin=0, vmax=1)
    if show_changes:
        for (r, c), kind in spec.changed_cells:
            is_now_open = (kind == "opened") == (phase == "post")
            if phase == "post" or kind == "closed":
                color = OPEN_NEW if (kind == "opened" and phase == "post") else (CLOSED if (kind == "closed" and phase == "post") else None)
                if color:
                    ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, color=color, zorder=3))
        # in the pre phase mark doors that WILL close
        if phase == "pre":
            for (r, c), kind in spec.changed_cells:
                if kind == "closed":
                    ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, ec=CLOSED, lw=2, zorder=3))
    sr, sc = spec.start
    ax.text(sc, sr, "S", ha="center", va="center", color="#1b5e8a", fontweight="bold", zorder=5)
    for (r, c), val in goals:
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, color="#f2b705", zorder=4))
        if label_goals:
            ax.text(c, r, f"{val:g}", ha="center", va="center", fontsize=7, fontweight="bold", zorder=5)
    _grid_axes(ax, walls.shape)
    ax.set_title(title or f"{spec.name} - {phase}-change", fontsize=10)
    return ax


def plot_task_overview(spec, figsize=(9.5, 4.8)):
    """Side-by-side pre / post change layouts (+ reward values)."""
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    plot_maze(spec, "pre", axs[0], title="before change (training)")
    plot_maze(spec, "post", axs[1], title="after change")
    fig.suptitle(spec.name.replace("_", " "), fontweight="bold")
    fig.tight_layout()
    return fig


def plot_heatmap(env, values, ax=None, cmap="viridis", title=None, log=False, colorbar=True, mark=None):
    """Per-state values on the maze (walls blank)."""
    ax = ax or plt.subplots(figsize=(5, 5))[1]
    g = env.to_grid(values)
    if log:
        g = np.log10(np.maximum(g, 1e-12))
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad(WALL)
    im = ax.imshow(g, cmap=cm, interpolation="nearest")
    if mark is not None:
        r, c = env.cell_of(mark)
        ax.plot(c, r, "r*", ms=11, mec="white")
    _grid_axes(ax, g.shape)
    if title:
        ax.set_title(title, fontsize=10)
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    return ax


def plot_policy(env, Q_or_policy, ax=None, title=None, background=None, every=1):
    """Arrows of the greedy action per open state. Accepts a Q table (S,A) or a policy vector (S,)."""
    ax = ax or plt.subplots(figsize=(6, 6))[1]
    X = np.asarray(Q_or_policy)
    pol = X if X.ndim == 1 else X.argmax(axis=1)
    bg = env.to_grid(background) if background is not None else None
    cm = plt.get_cmap("Greys").copy()
    cm.set_bad(WALL)
    base = np.where(env.walls, np.nan, 0.0) if bg is None else bg
    ax.imshow(base, cmap=cm if bg is None else "viridis", interpolation="nearest", vmin=-0.5 if bg is None else None, vmax=1 if bg is None else None)
    om = env.open_mask()
    for s in np.flatnonzero(om):
        r, c = env.cell_of(s)
        if env.terminal[s] or (r % every) or (c % every):
            continue
        dr, dc = MOVES[int(pol[s])]
        ax.arrow(c, r, 0.35 * dc, 0.35 * dr, head_width=0.25, head_length=0.2, fc="k", ec="k", lw=0.4)
    for g in env.goal_states:
        r, c = env.cell_of(g)
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, color="#f2b705", zorder=4))
    _grid_axes(ax, env.maze.shape)
    if title:
        ax.set_title(title, fontsize=10)
    return ax


def plot_path(env, policy, ax, color="tab:red", max_len=400):
    """Most-likely path (no slip) of a deterministic policy from the start."""
    s = env.start_state
    pts = [env.cell_of(s)]
    for _ in range(max_len):
        if env.terminal[s]:
            break
        s2 = int(env._nxt[s, int(policy[s])])
        if s2 == s:
            break
        s = s2
        pts.append(env.cell_of(s))
    pts = np.array(pts)
    ax.plot(pts[:, 1], pts[:, 0], "-", color=color, lw=2, alpha=0.85, zorder=6)
    return len(pts) - 1


def plot_sr_fields(env, M, states, ncols=4, figsize=None, titles=None):
    """SR row of selected states as heat maps ('place fields')."""
    n = len(states)
    nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize or (3.2 * ncols, 3.2 * nrows), squeeze=False)
    for ax in axs.ravel():
        ax.axis("off")
    for ax, s in zip(axs.ravel(), states):
        ax.axis("on")
        plot_heatmap(env, np.asarray(M)[s], ax=ax, cmap="magma", mark=s, colorbar=False,
                     title=(titles[list(states).index(s)] if titles else f"SR row of state {s}"))
    fig.tight_layout()
    return fig
