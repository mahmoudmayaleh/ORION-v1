#!/usr/bin/env python3
"""Audit reward components to find the -1839 return source."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
from orion.actors.domain_actor import DomainActor
from orion.actors.policy import DomainPolicy
from orion.baselines.greedy_ffd import _run_greedy_ffd, GreedyConfig
from orion.config import TopologyConfig
from orion.mdo.coordinator import MDOConfig, MDOCoordinator
from orion.mdo.policy import MDOPolicy
from orion.mdo.types import PlanSummary
from orion.mdo.observation import build_mdo_observation, observation_to_tensor
from orion.sim.arrival_process import ArrivalProcess
from orion.sim.episode_runner import EpisodeRunner
from orion.substrate.topology_generator import generate_multi_domain_topology
from orion.types import InfrastructureTier

rng = np.random.default_rng(0)
sub = generate_multi_domain_topology(
    TopologyConfig(num_domains=3, nodes_per_domain=[8,10,12],
                   intra_link_density=0.4, inter_domain_links=4), rng)
delays = {}
for u, v, d in sub.graph.edges(data=True):
    sd = sub.graph.nodes[u]["domain_id"]
    dd = sub.graph.nodes[v]["domain_id"]
    if sd != dd:
        delays[(sd, dd)] = min(delays.get((sd, dd), float("inf")), d["propagation_delay"])

def _plan(sr, substrate):
    r = _run_greedy_ffd(substrate, sr, GreedyConfig())
    if not r.feasible or r.plan is None:
        return None
    vnf_ids, tiers, doms = [], [], []
    for v in sr.vnfs:
        nid = r.plan.vnf_placements[v.vnf_id]
        d = int(nid.split("n")[0][1:])
        vnf_ids.append(v.vnf_id)
        tiers.append(InfrastructureTier(substrate.graph.nodes[nid]["tier"]))
        doms.append(d)
    return PlanSummary(
        vnf_ids=vnf_ids, required_tiers=tiers, suggested_domains=doms,
        cpu_demands=[v.cpu_demand for v in sr.vnfs],
        ram_demands=[v.ram_demand for v in sr.vnfs],
        vcrs=[v.vcr for v in sr.vnfs],
        bw_demands=[f.bandwidth_demand for f in sr.flow_edges])

obs = build_mdo_observation(sub, PlanSummary(
    vnf_ids=["v0","v1"], required_tiers=[InfrastructureTier.MEC]*2,
    suggested_domains=[0,1], cpu_demands=[1.0]*2, ram_demands=[1.0]*2,
    vcrs=[1.0]*2, bw_demands=[10.0]))
obs_dim = observation_to_tensor(obs, max_vnfs=10).shape[0]

torch.manual_seed(0)
np.random.seed(0)
policy = MDOPolicy(obs_dim=obs_dim, num_domains=3, max_vnfs=10, hidden_dim=128, num_layers=2)
actors = {d: DomainActor(d, DomainPolicy(backbone_type="mlp", hidden_dim=64)) for d in range(3)}
coord = MDOCoordinator(policy, actors, MDOConfig(n_part=3))
rng2 = np.random.default_rng(1)
ap = ArrivalProcess(sub, 500, 4.0, 0.02, rng2)
ap.generate()
runner = EpisodeRunner(sub, ap, coord, delays, plan_builder=_plan)
runner.reset()
ep = runner.run_episode(mdo_mode="sample")

print("=== REWARD COMPONENT AUDIT ===")
print()

# Admitted slices
admit_rewards = []
reject_rewards = []
for r in ep.mdo_results:
    rc = r.reward
    if r.admitted:
        admit_rewards.append(rc.total)
        print(f"ADMIT cost={r.total_cost:.1f} | adm={rc.admission:.0f} eff={rc.efficiency:.1f} "
              f"qual={rc.quality_shaping:.3f} trial={rc.trial_penalty:.1f} => total={rc.total:.1f}")
    else:
        reject_rewards.append(rc.total)

print()
print(f"Admitted: {len(admit_rewards)}")
if admit_rewards:
    print(f"  Admit rewards: min={min(admit_rewards):.1f} max={max(admit_rewards):.1f} "
          f"mean={sum(admit_rewards)/len(admit_rewards):.1f}")
print(f"Rejected: {len(reject_rewards)}")
if reject_rewards:
    print(f"  Reject rewards: min={min(reject_rewards):.1f} max={max(reject_rewards):.1f} "
          f"mean={sum(reject_rewards)/len(reject_rewards):.1f}")

# The real question: what fraction of admissions have NEGATIVE total reward?
neg_admits = [r for r in admit_rewards if r < 0]
print(f"\nAdmissions with NEGATIVE reward: {len(neg_admits)}/{len(admit_rewards)}")
if neg_admits:
    print(f"  Range: {min(neg_admits):.1f} to {max(neg_admits):.1f}")

# Now check the per-transition rewards (what goes into the buffer)
trans_rewards = [t.terminal_reward for t in ep.rollout.mdo]
print(f"\nPer-transition rewards (buffer): n={len(trans_rewards)}")
print(f"  min={min(trans_rewards):.1f} max={max(trans_rewards):.1f} "
      f"mean={sum(trans_rewards)/len(trans_rewards):.1f}")

# Breakdown by reward sign
pos = [r for r in trans_rewards if r > 0]
neg = [r for r in trans_rewards if r < 0]
zero = [r for r in trans_rewards if r == 0]
print(f"  positive: {len(pos)}, negative: {len(neg)}, zero: {len(zero)}")
if pos:
    print(f"  positive range: {min(pos):.1f} to {max(pos):.1f}")
if neg:
    print(f"  negative range: {min(neg):.1f} to {max(neg):.1f}")

# GAE returns to see the -1839
from orion.training.gae import compute_gae
rewards_t = torch.tensor(trans_rewards, dtype=torch.float32)
# Use constant critic value of 0 to see raw returns
values_t = torch.zeros(len(trans_rewards) + 1, dtype=torch.float32)
dones_t = torch.tensor([1.0 if t.committed else 0.0 for t in ep.rollout.mdo], dtype=torch.float32)
advantages, returns = compute_gae(rewards_t, values_t, dones_t, gamma=0.99, lam=0.95)
print(f"\nGAE returns (V=0 baseline): min={returns.min():.1f} max={returns.max():.1f} "
      f"mean={returns.mean():.1f} std={returns.std():.1f}")

# Check done flags
committed_count = sum(1 for t in ep.rollout.mdo if t.committed)
print(f"\nDone flags: {committed_count}/{len(ep.rollout.mdo)} committed")
print(f"  (Should be ~num_arrivals, each arrival ends with committed=True)")
