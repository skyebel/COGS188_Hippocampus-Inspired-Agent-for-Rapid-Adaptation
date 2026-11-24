"""Training protocol constants (from the proposal) and agent defaults."""
import numpy as np

# --- protocol: train until the agent reaches a goal in >= 90% of the last 20 episodes
#     (at most 500 episodes); optionally apply the change and continue for 200 episodes.
MAX_TRAIN_EPISODES = 500
CRITERION_WINDOW = 20
CRITERION_SUCCESS = 0.90
POST_EPISODES = 200
REPLAY_BUDGETS = [0, 5, 20, 50]            # backups per real step (ablation values)

# Proposal: tune hyperparameters on ~10 seeds, report final results on a SEPARATE set of 50+ seeds.
TUNE_SEEDS = list(range(10))
EVAL_SEEDS = list(range(1000, 1050))

COMMON = dict(gamma=0.99, epsilon=0.1)
# SR agents use the proposal's SARSA-TD update of the reward weights (reward_mode='td').
# alpha_sr is small because the entries of H scale like 1/(1-gamma) at gamma=0.99.
SR_DEFAULTS = dict(alpha_sr=0.01, alpha_td=0.002, reward_mode="td")
AGENT_DEFAULTS = {
    "q_learning":          dict(alpha=0.1),
    "dyna_q":              dict(alpha=0.1, replay_budget=20),
    "model_based_planner": dict(forget=0.95, plan_every=10),
    "sr_td":               dict(**SR_DEFAULTS),
    "sr_dyna":             dict(**SR_DEFAULTS, replay_budget=20),
    "prioritized_q":       dict(alpha=0.1, replay_budget=20),
    "sr_prioritized":      dict(**SR_DEFAULTS, replay_budget=20),
}


def agent_kwargs(agent, overrides=None):
    kw = dict(COMMON); kw.update(AGENT_DEFAULTS[agent])
    if overrides: kw.update(overrides)
    return kw


def split_seed(seed):
    """One integer seed -> independent (environment, agent) random streams."""
    e, a = np.random.SeedSequence(int(seed)).generate_state(2)
    return int(e), int(a)
