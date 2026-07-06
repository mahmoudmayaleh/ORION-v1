"""Per-agent rollout storage for the A3 hybrid env interface.

The top-level Gymnasium env (`sim/env.py`) is single-agent from the outside —
one step per slice arrival, one composite reward per step. Underneath, every
arrival generates multiple agent-level transitions:

    - MDO transitions: one per partition retry attempt (up to N_part).
      All retries within an arrival receive the same terminal reward (SMDP
      credit assignment per v6.2 §6.5).
    - Domain actor transitions: one per VNF placement / link routing decision
      inside each involved domain, per partition trial.

This module just stores those transitions. The Phase 5 MAPPO loop will read
them, compute GAE / advantages, and run the PPO update. Storage only — no
gradient logic here.

Shape of a "transition" is intentionally minimal: anything more elaborate
(per-step value bootstraps, masks, info dicts) is out of scope for Phase 1
because Phase 5 owns the optimiser side and will know what it needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class MDOTransition:
    """One MDO partition decision within one arrival.

    Multiple of these may share `terminal_reward` and `request_id` when
    the MDO retried within the arrival (SMDP credit assignment).
    """

    request_id: str
    trial_index: int
    obs: torch.Tensor                 # the MDO observation tensor at decision time
    action: list[int]                 # per-VNF domain assignment
    log_probs: torch.Tensor           # per-VNF log-probability of the chosen action
    entropy: float                    # mean entropy across VNF slots
    value_estimate: float             # V^MDO_ψ at decision time
    terminal_reward: float            # shared across all trials of this arrival
    committed: bool                   # True iff the arrival ended in COMMIT
    tier_mask: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    num_vnfs: int = 0
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class DomainActorTransition:
    """One domain-actor episode (per-fragment) within one partition trial.

    Stores per-VNF-step inputs for PPO re-evaluation (CTDE) alongside the
    slice-level summary the centralised critic needs.
    """

    request_id: str
    trial_index: int
    domain_id: int
    log_probs: torch.Tensor           # concatenated log-probs from the actor
    entropy: float
    accepted: bool                    # z^m
    intra_delay: float
    resource_cost: float
    terminal_reward: float            # broadcast from the arrival
    steps: list = field(default_factory=list)  # list[ActorStepRecord] from actors/types
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiAgentRollout:
    """Episode-scope rollout for the MDO and all domain actors.

    Append-only during rollout; `clear()` between training rounds. The Phase 5
    MAPPO loop will iterate over `.mdo` and `.domain_actor[domain_id]` to
    compute per-agent advantages.
    """

    mdo: list[MDOTransition] = field(default_factory=list)
    domain_actor: dict[int, list[DomainActorTransition]] = field(default_factory=dict)

    def append_mdo(self, transition: MDOTransition) -> None:
        self.mdo.append(transition)

    def append_domain_actor(self, transition: DomainActorTransition) -> None:
        self.domain_actor.setdefault(transition.domain_id, []).append(transition)

    def clear(self) -> None:
        self.mdo.clear()
        self.domain_actor.clear()

    @property
    def num_mdo_transitions(self) -> int:
        return len(self.mdo)

    @property
    def num_domain_actor_transitions(self) -> int:
        return sum(len(xs) for xs in self.domain_actor.values())

    def __len__(self) -> int:
        return self.num_mdo_transitions
