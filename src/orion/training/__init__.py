"""Phase 5 — CTDE training loop (MAPPO) and BC pretraining.

This package owns everything related to *learning*: the centralised critic
V_φ(s_t), the GAE/PPO machinery, the KL-prior schedule that pulls the MDO
toward Agent B's suggestion, and the behaviour-cloning warm-start of the
domain actors. The actors and the MDO policy themselves live in
`orion.actors` and `orion.mdo` and are imported here as black boxes.

Modules:
    config           MAPPOConfig dataclass with documented defaults.
    gae              Generalised Advantage Estimation.
    kl_schedule      β_t schedules for the KL-prior term (linear / constant / off).
    global_state     Encode V_φ's input from substrate + current arrival.
    critic           CentralisedCritic V_φ(s_t). Training-only; discarded at inference.
    buffer           PPO-ready rollout buffer (extends MultiAgentRollout).
    bc_dataset       Demonstrations from greedy FFD, seed-recorded and hashed.
    bc_pretrain      BC training loop for domain actors.
    ppo_update       PPO clipped surrogate + KL + value loss with separate scalars.
    mappo_trainer    Main joint training loop.
"""
