"""Train an agent in a maze following the proposal's protocol.

  1. train in the original maze until the agent reaches a goal in >= 90% of its last 20
     episodes (at most 500 episodes);
  2. optionally apply the environmental change and continue for `n_post` episodes (proposal: 200).

Logged per episode: steps, total reward, whether a goal was reached (and which), replay updates,
run time. Optionally every transition (state, action, reward, next state, done) is stored.
One seed controls the agent's choices and the environment's slips (independent streams)."""
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from agents import AGENTS
from envs import make_env
from .reference import ExactModel
from .config import (CRITERION_SUCCESS, CRITERION_WINDOW, MAX_TRAIN_EPISODES, POST_EPISODES,
                     agent_kwargs, split_seed)


@dataclass
class TrainResult:
    env: object
    agent: object
    episodes: pd.DataFrame
    meta: dict
    visits: dict = field(default_factory=dict)       # state visit counts per phase
    steps: dict = None                               # transition log (if log_steps=True)
    replay: pd.DataFrame = None                      # one row per replay backup (if log_replay), cols = REPLAY_LOG_FIELDS + 'phase'
    snapshots: dict = field(default_factory=dict)    # {'pre'|'post': {'Q': (S,A), 'M': (S,S) state-level SR (SR agents only)}}
    reference: dict = field(default_factory=dict)    # J_star, J_rand, optimal_path of the post-change maze (see reference.py)


def snapshot(agent):
    """Learned quantities at the end of a phase: Q table and, for SR agents, the state-level SR matrix M
    (rows = place-field-like maps; for SR shape / Moran's I / eigenvector analyses)."""
    out = {"Q": np.array(agent.q_table(), dtype=np.float32)}
    if hasattr(agent, "state_sr"):
        out["M"] = agent.state_sr().astype(np.float32)
    return out


def make_agent(name, env, overrides=None, seed=0, **flags):
    kw = agent_kwargs(name, overrides); kw.update(flags)
    return AGENTS[name](env.n_states, env.n_actions, seed=seed, **kw)


def run_episode(env, agent, visits=None, step_log=None, tag=(0, 0)):
    """One episode. The agent is told `done` only on true termination (goal), not on the 1000-step cap."""
    s, _ = env.reset(); agent.begin_episode(s)
    ret, n, goal, terminated, truncated = 0.0, 0, -1, False, False
    while not (terminated or truncated):
        a = agent.select_action(s)
        s2, r, terminated, truncated, _ = env.step(a)
        agent.update(s, a, r, s2, terminated)
        if visits is not None: visits[s2] += 1
        if step_log is not None: step_log.append((*tag, n, s, a, r, s2, terminated))
        ret += r; n += 1; s = s2
        if terminated: goal = s2
    agent.end_episode()
    return n, ret, goal, bool(terminated)


def train(task, agent_name, seed=0, scale="full", overrides=None, max_episodes=MAX_TRAIN_EPISODES,
          window=CRITERION_WINDOW, success=CRITERION_SUCCESS, n_post=0, slip=0.10, max_steps=1000,
          log_steps=False, verbose=False, eval_phases=("post",), log_replay=None, variant="default",
          **agent_flags) -> TrainResult:
    """eval_phases : phases after whose every episode the agent's policy is evaluated EXACTLY
                     (columns J_greedy, J_behavior, J_star, J_rand; NaN elsewhere). Needed for rho_e / recovery / AUC / regret.
       log_replay  : None (off) | 'post' | 'all' - record every replay backup (needed for H3 / replay-direction metrics).
                     Big (k x steps rows), so use for a subset of seeds. For the 'pre_move' context also pass replay_at_start=True.
       variant     : free-text label for ablation runs (e.g. 'k=5'); stored in every row so runs can be grouped later.
       Other agent flags / hyperparameter overrides: see make_agent / config.agent_kwargs."""
    log_replay = "all" if log_replay is True else (log_replay or None)
    env_seed, agent_seed = split_seed(seed)
    env = make_env(task, scale, slip=slip, max_steps=max_steps)
    env.reset(seed=env_seed)
    agent = make_agent(agent_name, env, overrides, seed=agent_seed, **agent_flags)
    rows, visits = [], {"pre": np.zeros(env.n_states), "post": np.zeros(env.n_states)}
    step_log, recent, met, prev_b = ([] if log_steps else None), deque(maxlen=window), False, 0
    t_all = time.perf_counter()
    snaps, ref, model, post_replay_row = {}, {}, None, 0
    for phase, n_eps in (("pre", max_episodes), ("post", n_post)):
        if phase == "post":
            snaps["pre"] = snapshot(agent)            # end of training, before the change
            agent._flush_replay_buf(); post_replay_row = sum(len(c) for c in agent._replay_chunks)
            env.set_phase("post")                     # the environmental change
            if n_post == 0: break
        agent.log_replay = log_replay == "all" or (log_replay == "post" and phase == "post")
        if phase in eval_phases:
            model = ExactModel(env); ref = model.reference()      # exact model of the maze in THIS phase
        else:
            model = None
        for ep in range(1, n_eps + 1):
            t0 = time.perf_counter()
            steps, ret, goal, reached = run_episode(env, agent, visits[phase], step_log, (int(phase == "post"), ep))
            row = dict(task=task, agent=agent_name, variant=variant, seed=seed, phase=phase, episode=ep, steps=steps,
                       reward=ret, reached_goal=reached, goal_state=goal,
                       replay_updates=agent.n_backups - prev_b, runtime_s=time.perf_counter() - t0)
            if model is not None:   # exact policy values after this episode (not counted in runtime_s)
                row.update(model.evaluate_agent(agent), J_star=ref["J_star"], J_rand=ref["J_rand"])
            rows.append(row)
            prev_b = agent.n_backups
            if verbose and ep % 25 == 0: print(f"  {phase} episode {ep}")
            if phase == "pre":
                recent.append(reached)
                if len(recent) == window and np.mean(recent) >= success:
                    met = True
                    break
    if n_post > 0: snaps["post"] = snapshot(agent)        # end of the post-change phase
    meta = dict(task=task, agent=agent_name, variant=variant, seed=seed, scale=scale, n_states=env.n_states,
                train_episodes=int((pd.DataFrame(rows).phase == "pre").sum()), criterion_met=met,
                total_replay_updates=agent.n_backups, total_runtime_s=time.perf_counter() - t_all,
                agent_kwargs={k: v for k, v in agent_kwargs(agent_name, overrides).items()})
    steps_out = None
    if log_steps:
        a = np.asarray(step_log, dtype=np.float64).T
        names = ("phase", "episode", "step", "state", "action", "reward", "next_state", "done")
        steps_out = {n: a[i] for i, n in enumerate(names)}
    replay = None
    if log_replay:
        replay = pd.DataFrame(agent.replay_log_arrays())
        replay["phase"] = np.where(np.arange(len(replay)) < post_replay_row, "pre", "post")   # rows are chronological
    return TrainResult(env, agent, pd.DataFrame(rows), meta, visits, steps_out, replay, snaps,
                       {k: ref[k] for k in ("J_star", "J_rand", "optimal_path")} if ref else {})
