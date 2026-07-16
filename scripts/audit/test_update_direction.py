#!/usr/bin/env python3
"""Verification battery — Phase 2, check 9: update-direction integration test.

One synthetic batch where action A has advantage +1 and action B has -1,
pushed through the REAL PPO update code (the MDO update lines of
wp7_runner.train_arm, scripts/wp7_runner.py:424-455, replicated verbatim
below with beta=0 and the same normalization/clipping constants the gate
uses). Assert: probability mass moves toward A after one update.

Uses the real MDOPolicy class, real evaluate_actions, real clip/entropy
constants (clip_eps=0.2, entropy_coef=0.01, lr=3e-3, update_epochs=4,
grad-clip 0.5).
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "src"))

from orion.mdo.policy import MDOPolicy

torch.manual_seed(0)

OBS_DIM = 16
M = 5          # domains
K = 1          # one VNF slot
CLIP_EPS = 0.2
ENT_COEF = 0.01
LR = 3e-3
EPOCHS = 4

policy = MDOPolicy(obs_dim=OBS_DIM, num_domains=M, max_vnfs=10,
                   hidden_dim=128, num_layers=2)
opt = torch.optim.Adam(policy.parameters(), lr=LR)

obs = torch.randn(OBS_DIM)
tier_mask = torch.ones(K, M, dtype=torch.bool)

ACTION_A, ACTION_B = 2, 4

# Old log-probs at collection time (policy before update)
with torch.no_grad():
    lp_a0, _, _ = policy.evaluate_actions(obs, tier_mask, torch.tensor([ACTION_A]), K)
    lp_b0, _, _ = policy.evaluate_actions(obs, tier_mask, torch.tensor([ACTION_B]), K)

batch = [
    {"action": ACTION_A, "adv": +1.0, "old_lp": lp_a0.detach()},
    {"action": ACTION_B, "adv": -1.0, "old_lp": lp_b0.detach()},
]

p_before = torch.softmax(
    policy.evaluate_actions(obs, tier_mask, torch.tensor([ACTION_A]), K)[2][0], dim=-1
).detach()

# ── Replicated verbatim from wp7_runner.train_arm (MDO PPO update), beta=0 ──
for _ in range(EPOCHS):
    epoch_loss = torch.tensor(0.0); cnt = 0
    for row in batch:
        action_i = torch.tensor([row["action"]], dtype=torch.long)
        adv_i = row["adv"]
        new_lp, new_ent, new_logits = policy.evaluate_actions(
            obs, tier_mask, action_i, K)
        ratio = torch.exp(new_lp.sum() - row["old_lp"].sum())
        step_loss = -torch.min(
            ratio * adv_i,
            torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_i)
        epoch_loss = epoch_loss + step_loss - ENT_COEF * new_ent
        cnt += 1
    if cnt > 0:
        opt.zero_grad(); (epoch_loss / cnt).backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5); opt.step()
# ── end replication ──

p_after = torch.softmax(
    policy.evaluate_actions(obs, tier_mask, torch.tensor([ACTION_A]), K)[2][0], dim=-1
).detach()

print(f"P(A={ACTION_A}) before={p_before[ACTION_A]:.4f} after={p_after[ACTION_A]:.4f}")
print(f"P(B={ACTION_B}) before={p_before[ACTION_B]:.4f} after={p_after[ACTION_B]:.4f}")

assert p_after[ACTION_A] > p_before[ACTION_A], "FAIL: mass did not move toward +adv action"
assert p_after[ACTION_B] < p_before[ACTION_B], "FAIL: mass did not move away from -adv action"
print("PASS: probability mass moves toward the +1-advantage action through the real update code")
