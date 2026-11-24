"""Hippocampus-inspired RL agents (COGS 188 project: Rat-Brained).

    from agents import QLearningAgent, SRPrioritizedAgent, AGENTS
"""
from .base import BaseAgent
from .q_learning import QLearningAgent
from .dyna_q import DynaQAgent
from .model_based_planner import ModelBasedPlanner
from .sr_td import SRTDAgent
from .sr_dyna import SRDynaAgent
from .prioritized_q import PrioritizedQAgent
from .sr_prioritized import SRPrioritizedAgent
from .replay import (classify_replay_direction, classify_replay_context, epsilon_greedy_probs,
                     prioritized_replay, REPLAY_LOG_FIELDS)

AGENTS = {
    "q_learning": QLearningAgent,
    "dyna_q": DynaQAgent,
    "model_based_planner": ModelBasedPlanner,
    "sr_td": SRTDAgent,
    "sr_dyna": SRDynaAgent,
    "prioritized_q": PrioritizedQAgent,
    "sr_prioritized": SRPrioritizedAgent,
}

__all__ = ["BaseAgent", "AGENTS", "classify_replay_direction", "classify_replay_context", "epsilon_greedy_probs",
           "prioritized_replay", "REPLAY_LOG_FIELDS",
           *[c.__name__ for c in AGENTS.values()]]
