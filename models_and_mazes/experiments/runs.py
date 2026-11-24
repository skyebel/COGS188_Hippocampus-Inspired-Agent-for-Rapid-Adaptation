"""Save / load / batch-run training runs so the analysis team gets one standard on-disk format.

    root/<task>/<agent>[__<variant>]/seed<N>/
        episodes.csv    per-episode log (+ exact J_greedy / J_behavior / J_star / J_rand in the post phase)
        meta.json       run settings, criterion_met, total replay updates, J_star, J_rand, optimal_path
        snapshots.npz   pre_Q, pre_M, post_Q, post_M   (M = state-level SR, SR agents only)
        replay.npz      replay-event log incl. 'phase'            (only if log_replay was used)
        steps.npz       every transition                          (only if log_steps was used)

    from experiments.runs import run_grid, load_runs
    run_grid(TASKS, AGENTS, EVAL_SEEDS, "runs", n_post=200)        # resumable: finished runs are skipped
    df = load_runs("runs")                                        # all episode logs in one DataFrame
"""
import json, os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .train import train
from .config import AGENT_DEFAULTS


def run_dir(root, task, agent, seed, variant="default"):
    return os.path.join(root, task, agent if variant == "default" else f"{agent}__{variant}", f"seed{seed}")


def save_run(res, root):
    m = res.meta
    d = run_dir(root, m["task"], m["agent"], m["seed"], m["variant"]); os.makedirs(d, exist_ok=True)
    json.dump({**m, **res.reference}, open(os.path.join(d, "meta.json"), "w"), indent=2, default=str)
    np.savez_compressed(os.path.join(d, "snapshots.npz"),
                        **{f"{ph}_{k}": v for ph, sn in res.snapshots.items() for k, v in sn.items()})
    if res.replay is not None:
        np.savez_compressed(os.path.join(d, "replay.npz"), **{c: res.replay[c].to_numpy() if res.replay[c].dtype.kind in "biuf" else res.replay[c].to_numpy().astype(str)
                                                for c in res.replay})
    if res.steps is not None:
        np.savez_compressed(os.path.join(d, "steps.npz"), **res.steps)
    res.episodes.to_csv(os.path.join(d, "episodes.csv"), index=False)     # written last = "run finished" marker
    return d


def _one(args):
    task, agent, seed, variant, overrides, root, kw = args
    if os.path.exists(os.path.join(run_dir(root, task, agent, seed, variant), "episodes.csv")):
        return                                                            # already done
    save_run(train(task, agent, seed, overrides=overrides, variant=variant, **kw), root)


def run_grid(tasks, agents, seeds, root, n_post=200, variants=None, n_workers=None, **train_kw):
    """Run every task x agent x seed (x variant) in parallel processes and save each run.
    variants: {label: overrides dict}, e.g. {'k=5': dict(replay_budget=5), 'k=50': dict(replay_budget=50)}
              (default: just the default hyperparameters). Tip: export OMP_NUM_THREADS=1 first (many workers x BLAS threads).
    Extra keyword args go to train() (e.g. log_replay='post', replay_at_start=True for the H3 subset of seeds)."""
    variants = variants or {"default": None}
    # an override only applies to agents that have that hyperparameter (e.g. replay_budget is skipped for q_learning)
    ok = lambda a, ov: ov is None or set(ov) <= set(AGENT_DEFAULTS[a]) | {"gamma", "epsilon"}
    jobs = [(t, a, s, v, ov, root, dict(n_post=n_post, **train_kw))
            for t in tasks for a in agents for s in seeds for v, ov in variants.items() if ok(a, ov)]
    with ProcessPoolExecutor(n_workers) as ex:
        list(ex.map(_one, jobs))


def load_run(d):
    """Everything saved for one run directory as a dict (episodes DataFrame, meta, snapshots, replay, steps)."""
    out = dict(episodes=pd.read_csv(os.path.join(d, "episodes.csv")), meta=json.load(open(os.path.join(d, "meta.json"))))
    for key, fn in (("snapshots", "snapshots.npz"), ("replay", "replay.npz"), ("steps", "steps.npz")):
        out[key] = dict(np.load(os.path.join(d, fn))) if os.path.exists(os.path.join(d, fn)) else None
    return out


def load_runs(root, tasks=None, agents=None):
    """Concatenate episodes.csv of every finished run under root (columns: task, agent, variant, seed, phase, ...)."""
    dfs = []
    for dp, _, files in os.walk(root):
        if "episodes.csv" in files:
            df = pd.read_csv(os.path.join(dp, "episodes.csv"))
            if (tasks is None or df.task[0] in tasks) and (agents is None or df.agent[0] in agents):
                dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
