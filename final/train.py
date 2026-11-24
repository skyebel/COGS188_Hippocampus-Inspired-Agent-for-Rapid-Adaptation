#!/usr/bin/env python
"""Command line: python train.py --task transition_revaluation --agent sr_prioritized --seed 0 --post 200 [--log-replay post] [--out runs]
For many seeds / agents / ablation variants use experiments.runs.run_grid instead."""
import argparse
from experiments.train import train
from experiments.runs import save_run
from experiments.config import POST_EPISODES

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="transition_revaluation")
ap.add_argument("--agent", default="sr_prioritized")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--post", type=int, default=0, help=f"episodes after the change (proposal: {POST_EPISODES})")
ap.add_argument("--max-episodes", type=int, default=500)
ap.add_argument("--log-steps", action="store_true")
ap.add_argument("--log-replay", choices=["post", "all"], default=None, help="save every replay backup (H3 metrics)")
ap.add_argument("--replay-at-start", action="store_true", help="extra replay before the first move of each episode (H3 'pre_move')")
ap.add_argument("--out", default="runs")
a = ap.parse_args()
res = train(a.task, a.agent, a.seed, n_post=a.post, max_episodes=a.max_episodes, log_steps=a.log_steps,
            log_replay=a.log_replay, replay_at_start=a.replay_at_start, verbose=True)
print("saved to", save_run(res, a.out), res.meta)
