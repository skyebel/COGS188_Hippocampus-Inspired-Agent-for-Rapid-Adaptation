# Rat-Brained: agents trained in the proposed mazes

    agents/        7 agents (Q-learning, Dyna-Q, model-based planner, SR-TD, SR-Dyna, prioritized-Q, SR + prioritized replay)
    envs/          3 stochastic multi-room mazes (29x29, 738 states, 10% slip): reward revaluation,
                   transition revaluation, Tolman detour (+ shortcut variant); Gymnasium interface
    experiments/   config.py   protocol constants, agent defaults, TUNE_SEEDS / EVAL_SEEDS
                   train.py    training runner (+ per-episode exact policy values, replay log, SR/Q snapshots)
                   reference.py exact J_star / J_rand / policy values (ExactModel)
                   runs.py     run_grid (many seeds in parallel), save_run, load_runs, load_run
    viz/           maze plots
    train.py       command line training
    Rat_Brained_Agents_in_Mazes.ipynb

Protocol: train until a goal is reached in >= 90% of the last 20 episodes (max 500), optionally apply the change and
continue for 200 episodes. One seed controls the agent and the environment slips.

    from experiments.train import train
    res = train("transition_revaluation", "sr_prioritized", seed=0, n_post=200)
    res.episodes   # per-episode log (steps, reward, goal reached, replay updates, run time)
    res.agent, res.env

## For the analysis team

No metrics or statistics are computed here; the data needed for them is. After `train(..., n_post=200)`:

| Need | Where |
|---|---|
| J_e (greedy), behavior value (regret), J_star, J_rand after every post-change episode | `res.episodes` columns `J_greedy, J_behavior, J_star, J_rand` (exact, discounted with the task gamma, no 1000-step cap) |
| steps / reward / goal / replay updates / run time per episode | `res.episodes` (`phase` = pre / post, `variant` labels ablations) |
| replay events (direction, context, EVB, need, gain) | `train(..., log_replay='post'|'all')` -> `res.replay` (has `phase`); label with `agents.classify_replay_direction / classify_replay_context`. For the `pre_move` context also pass `replay_at_start=True` |
| SR matrix / Q at end of each phase | `res.snapshots['pre'|'post']['M'|'Q']` (`M` = state-level SR, SR agents only) |
| Moran's I weights | `res.env.adjacency()`; map an SR row onto the maze with `res.env.to_grid(M[s])` |
| 'replay content': is replay on the task path? | `res.reference['optimal_path']` (post-change optimal path, list of states) |
| exact values for other policies / mazes | `ExactModel(env).value(pi)`, `.reference()` (build a fresh one after `env.set_phase('post')`) |

Many runs: `run_grid(TASKS, AGENTS, EVAL_SEEDS, 'runs', n_post=200, variants={'k=5': dict(replay_budget=5)})`
(resumable; `export OMP_NUM_THREADS=1` first), then `load_runs('runs')` for one DataFrame, or `load_run(dir)` for a
single run's episodes, meta, snapshots, replay and steps. Tune on `TUNE_SEEDS`, report on `EVAL_SEEDS`.
Full replay logs are large (k x steps rows): use them for a subset of seeds.
