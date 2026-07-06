"""MAPPO training configuration.

Defaults follow Huang et al. 2022 ("The 32 Implementation Details of PPO")
and the CleanRL `ppo.py` reference. Anything that differs from CleanRL is
documented inline.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MAPPOConfig:
    """Hyperparameters for the joint CTDE training loop.

    Logging note: the trainer must emit `kl_prior_term` and `entropy_bonus`
    as separate scalars in the metrics dict, never a fused "regularisation"
    term. Both shape exploration in different ways and need to be
    disentangled when diagnosing convergence.
    """

    # ── Optimisation ────────────────────────────────────────────────────
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    minibatch_size: int = 64
    update_epochs: int = 4
    max_grad_norm: float = 0.5
    # Anneal the learning rate linearly across total_timesteps.
    anneal_lr: bool = True

    # ── PPO core ────────────────────────────────────────────────────────
    clip_eps: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    # Clip the value loss too (CleanRL detail). Reduces value spikes.
    clip_value_loss: bool = True

    # ── GAE ─────────────────────────────────────────────────────────────
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # ── KL prior toward Agent B ────────────────────────────────────────
    # Linear decay (Choice D1). Ablation axis: beta_zero / beta_constant /
    # beta_linear, all sharing the same MAPPO core so the prior is the only
    # thing changing. NOT an ablation against target-KL adaptive — that's
    # a different regulariser (it pins the policy at a fixed divergence,
    # which is the opposite of "outgrow the prior").
    kl_beta_initial: float = 1.0
    kl_beta_final: float = 0.0
    kl_beta_decay_steps: int = 500_000

    # ── Rollout collection ──────────────────────────────────────────────
    # Choice C1: single-process sequential rollout. The bottleneck is
    # Agent B latency on cache misses, not env step cost; VectorEnv would
    # not help here. If throughput becomes a problem, the lever is LLM
    # batching/async, not VectorEnv.
    num_envs: int = 1
    steps_per_rollout: int = 1024
    total_timesteps: int = 1_000_000

    # ── BC pretraining (Phase 0 of Phase 5) ─────────────────────────────
    # Choice B1: cache demonstrations to disk with fixed seeds and a
    # dataset hash, so the paper can state exactly what BC trained on.
    bc_num_scenarios: int = 5000
    bc_epochs: int = 10
    bc_lr: float = 1e-3
    bc_entropy_coef: float = 0.01  # prevents collapse onto deterministic greedy
    bc_seed: int = 42
    bc_dataset_path: str = "data/bc_demonstrations.pt"

    # ── Centralised critic (Choice A1) ──────────────────────────────────
    # Flat MLP over canonical-ordered global state. The critic is
    # training-only and discarded at inference. The MDO and the critic
    # MUST consume domains in the same canonical order (tier_type, domain_id).
    critic_hidden_dim: int = 256
    critic_num_layers: int = 3

    # ── Logging ─────────────────────────────────────────────────────────
    log_interval: int = 10  # rollouts between log emissions
    checkpoint_interval: int = 50

    # ── Seed ────────────────────────────────────────────────────────────
    seed: int = 0
