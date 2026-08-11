"""§Y.2/§Y.3 — offered load as the difficulty axis.

The pre-§Y setup had ARRIVAL_RATE=4.0 with SERVICE_RATE=0.02, so the mean slice
lifetime was 50 time units against a ~25 time unit episode. Essentially nothing
departed: every episode was "pour slices in until the substrate fills", with no
steady state and no load knob. Difficulty came from a topology label instead.

Here difficulty is offered load. Arrivals are Poisson(lambda), lifetimes are
Exp(mu) with mu FIXED, resources are released on departure (`ArrivalProcess`
already emits the DEPARTURE events; the callers must honour them), and lambda is
the only thing that moves. Per the supervisor's note: vary the arrival rate, not
the lifetime.

Load is reported two ways, because they answer different questions:

  * `offered_load_fraction` -- offered load as a fraction of substrate capacity.
    Depends only on the workload and the infrastructure, NOT on which approach is
    running, so it is the honest x-axis for every figure. This is the quantity the
    supervisor asked for ("the load w.r.t capacity: 20%, 40%").
  * Resulting acceptance under a reference policy -- what actually makes a level
    hard. Two substrates at equal offered load are not equally difficult if their
    tier structure binds differently.

The four levels are CALIBRATED, not guessed, and this module refuses to hand out
uncalibrated ones: see `get_level`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from orion.sim.arrival_process import ArrivalProcess, SliceFactory
from orion.substrate.graph_model import SubstrateNetwork

# --------------------------------------------------------------------------
# Committed workload constants (§Y.2). Changing any of these is an amendment.
# --------------------------------------------------------------------------

#: Exponential lifetime rate. Mean lifetime 20 time units against a 200-arrival
#: episode, so departures are live and the occupancy process reaches steady state.
SERVICE_RATE = 0.05

#: Arrivals per episode.
#:
#: AMENDED 2026-07-29: 200 -> 2000, forced by the §Y.3 calibration. Offered load
#: is A = lambda/mu concurrent slices, but an N-arrival episode can never hold
#: more than N slices, and churn only exists if the ARRIVAL span covers several
#: mean lifetimes. At N=200 the ladder was unreachable: acceptance floored at
#: 0.72 and never approached the L3/L4 targets, only 2 of 8 sweep points were
#: valid steady-state measurements, and at the top of the sweep the arrival span
#: was 0.35 lifetimes with A/N = 2.7 -- the episode was "fill until arrivals run
#: out", not a queue. At N=2000 every sweep point is valid and acceptance falls
#: monotonically 0.87 -> 0.45.
#:
#: 2000 (user directive 2026-07-31). Briefly 3000 to let the ladder reach
#: rho = 2.00; at 2000 the top rung fails A/N <= 0.25 and is not a steady-state
#: measurement, so L4 is taken from the highest rho that IS one and the
#: acceptance range is correspondingly narrower. The targets below are set
#: against the range this N can actually produce.
#: Superseded note: 2000 -> 3000 was forced by the L4 offered load. The ladder now
#: reaches rho = 2.00, i.e. A = 528 concurrent slices, and an N-arrival episode can
#: never hold more than N of them: at N=2000 the top rung failed A/N <= 0.25 and was
#: not a steady-state measurement, so L4 had to be taken from rho = 1.52 instead and
#: the bottom of the achievable range was unreachable.
#:
#: The frozen levels below assume this value. `A/N <= 0.25` is asserted in the
#: tests so the two cannot drift apart again.
NUM_ARRIVALS = 2000

#: Arrivals discarded from every episode before metrics are computed. The
#: substrate starts empty, so early arrivals see a network that the steady-state
#: process never revisits. This is a property of the queueing system and applies
#: to EVERY approach, learned or not, not only to the RL ones. The separate
#: question of discarding early RL experience is handled by the entropy schedule
#: in the trainer and is deliberately not conflated with this.
WARMUP_ARRIVALS = 400

#: Seeds. 5 minimum everywhere (user directive 2026-07-28); nothing runs at 3.
SEEDS = (42, 43, 44, 45, 46)

#: Bracket for the §Y.3 calibration sweep, expressed in OFFERED LOAD, not in
#: lambda. A lambda bracket is not portable across substrate sizes: measured on
#: the Large reference, lambda=4.0 is only rho=0.30, so the pre-§Y bracket of
#: (0.5, 4.0) -- written for the 32-node substrate -- never reaches saturation
#: and could not have produced the L3/L4 acceptance targets. Sweeping in rho and
#: converting per substrate makes the same protocol work unchanged at every size
#: in §Y.11, which is required since each size is recalibrated.
RHO_SWEEP = (0.10, 2.00)

#: Acceptance targets the four levels are pinned to.
#:
#: L1 was 0.95 and is now 0.85, because 0.95 is NOT REACHABLE by the reference
#: policy. Measured on the calibrated substrate, Plain tops out at 0.870 even at
#: rho = 0.10 with 21 concurrent slices and 14% MEC utilisation. The residual is
#: the QoS gate, and it is legitimate: colocation FFD packs by best CPU fit, which
#: maximises per-node utilisation and therefore the M/M/1 queueing term, so it
#: loses slices to delay that a delay-aware placer would keep. That gap is exactly
#: the headroom the learned approaches are supposed to exploit, so it must not be
#: tuned away -- but a level cannot be defined at an acceptance the policy that
#: defines it never reaches.
#:
#: The four targets now span the achievable range (0.87 .. 0.45) rather than an
#: assumed one. Levels remain pinned on the OUTCOME, not on rho.
#: RE-TARGETED 2026-07-31 for the §Y.1e substrate. The previous targets
#: (0.85/0.75/0.60/0.45) are NOT REACHABLE here: measured on the calibrated
#: substrate the reference policy tops out at 0.740 even at rho = 0.10 with 20
#: concurrent slices and 12% utilisation of the binding tier, so L1 and L2 both
#: selected the same lambda (1.32) and the ladder degenerated to three distinct
#: levels wearing four names.
#:
#: The residual is the QoS gate and it is legitimate rather than a defect to tune
#: away: colocation FFD packs by best CPU fit, which is delay-blind, so it loses
#: slices to the delay constraint that a delay-aware placer would keep. That gap is
#: precisely the headroom the learned approaches exist to exploit. What is NOT
#: legitimate is defining a level at an acceptance the policy that defines it never
#: reaches.
#:
#: The achievable range on this substrate is 0.74 down to 0.50, narrower than the
#: 0.87-0.45 of the superseded four-tier substrate, because merging MEC into edge
#: concentrates the workload on one tier and the XR chain now bounces between edge
#: and regional (37% of XR is delay-infeasible on an EMPTY substrate; see
#: scripts/diag_delay_budget.py). The acceptance axis is therefore compressed, but
#: the LOAD axis is not: rho spans 0.10 to 2.00 and mean concurrency 20 to 220.
ACCEPTANCE_TARGETS = {"L1": 0.72, "L2": 0.66, "L3": 0.60, "L4": 0.55}

#: The load level every learned stack trains at (§Y.4).
TRAINING_LEVEL = "L2"


@dataclass(frozen=True)
class LoadLevel:
    """One calibrated rung of the load ladder."""

    name: str
    arrival_rate: float
    service_rate: float = SERVICE_RATE
    #: Offered load as a fraction of substrate CPU capacity, at the reference size.
    rho_offered: float | None = None
    #: Acceptance the reference greedy policy achieved during calibration.
    plain_acceptance: float | None = None

    @property
    def erlangs(self) -> float:
        """A = lambda / mu, the expected number of concurrent slices."""
        return self.arrival_rate / self.service_rate

    def __str__(self) -> str:
        rho = "--" if self.rho_offered is None else f"{self.rho_offered:.3f}"
        acc = "--" if self.plain_acceptance is None else f"{self.plain_acceptance:.3f}"
        return (f"{self.name}: lambda={self.arrival_rate:.3f} A={self.erlangs:.1f} "
                f"rho_offered={rho} plain_acceptance={acc}")


#: FROZEN 2026-07-29 by scripts/calibrate_load_levels.py on the committed
#: substrate: 12 rho points, seeds {42..46}, instances {0,1,2}, N=2000 arrivals,
#: 180 Plain episodes. Every sweep point was a valid steady-state measurement
#: (A/N <= 0.25 and arrival span >= 3 mean lifetimes) and acceptance fell
#: monotonically with rho.
#:
#: Named §Y.3 expectations at freeze time:
#:   acceptance monotone in rho ................ PASS
#:   concurrent(L4)/concurrent(L1) >= 3 ........ 6.46  PASS
#:   binding-tier utilisation at L4 > 0.85 ..... 0.94 (regional_cloud)  PASS
#:
#: Changing any of these four values is a new amendment, not a tuning step.
#:
#: RE-FROZEN 2026-07-30 on the §Y.1d substrate (lateral intra-domain links at 0.30,
#: star plus lateral inter-domain adjacencies) under the ratified acceptance
#: definition (admitted / total GENERATED, whole episode, no warm-up window).
#: 12 rho points, seeds {42..46}, instances {0,1,2}, N=2000, 180 Plain episodes,
#: 179 s wall. Every sweep point was a valid steady-state measurement and all three
#: named expectations passed (monotone; concurrency ratio 5.39; binding-tier
#: utilisation 0.96 on regional_cloud).
#:
#: `plain_acceptance` is the ratified metric. The steady-state value is recorded
#: next to it for reference: L1 0.851, L2 0.749, L3 0.603, L4 0.428, so the
#: transient contributes +0.002 / +0.011 / +0.024 / +0.043.
#:
#: Superseded 2026-07-29 freeze, for comparison only, do NOT reuse (windowed
#: metric, EXTRA_LINK_FRAC=0.15, pure star):
#:      L1 lambda=1.56  rho=0.131  acceptance=0.850
#:      L2 lambda=7.98  rho=0.673  acceptance=0.735
#:      L3 lambda=13.76 rho=1.160  acceptance=0.594
#:      L4 lambda=23.72 rho=2.000  acceptance=0.420
#: lambda rose at every level: the added links and capacity make the substrate
#: easier at equal lambda, so more offered load is needed to hit the same target.
#: FROZEN on the §Y.1e substrate (80 nodes, three tiers, heterogeneous domain
#: composition, partial mesh) by scripts/calibrate_load_levels.py:
#:   3000 arrivals, seeds [42, 43, 44, 45, 46], instances [0, 1, 2],
#:   12 rho points, 257.8 s wall.
#:
#: Acceptance monotone in rho: PASS (the script refuses to freeze otherwise, and
#: this table is written by it rather than by hand).
#:
#: Superseded ladders, for comparison only, do NOT reuse:
#:   2026-07-30, 100-node four-tier substrate:
#:     L1 2.0654 / L2 8.0608 / L3 13.8973 / L4 23.9597
#: FROZEN on the §Y.1e substrate (80 nodes, three tiers, heterogeneous domain
#: composition, partial mesh) by scripts/calibrate_load_levels.py at N=2000:
#:   12 rho points, seeds [42, 43, 44, 45, 46], instances [0, 1, 2], 167 s wall.
#: Monotone in rho, and every level has a distinct lambda -- the script refuses
#: to freeze otherwise, and writes this table itself rather than by hand.
#:
#: Superseded ladders, comparison only, do NOT reuse:
#:   2026-07-30, 100-node four-tier: L1 2.0654 L2 8.0608 L3 13.8973 L4 23.9597
#: FROZEN by scripts/freeze_load_levels.py from a §Y.3 calibration sweep:
#:   N=2000, 12 rho points, seeds [42, 43, 44, 45, 46], instances [0, 1, 2], 202 s wall.
#: Monotone in rho and every level has a distinct lambda; the script refuses
#: to write this table otherwise, and writes it itself rather than by hand.
#: Recalibrated after the Y.10 chain-truncation fix; the superseded ladder measured a workload in which 61% of arrivals were 2-VNF chains.
CALIBRATED_LEVELS: dict[str, LoadLevel] = {
    "L1": LoadLevel("L1", arrival_rate=1.0593, rho_offered=0.1000, plain_acceptance=0.7211),
    "L2": LoadLevel("L2", arrival_rate=7.1279, rho_offered=0.6729, plain_acceptance=0.6687),
    "L3": LoadLevel("L3", arrival_rate=12.2889, rho_offered=1.1601, plain_acceptance=0.6091),
    "L4": LoadLevel("L4", arrival_rate=16.1358, rho_offered=1.5232, plain_acceptance=0.5485),
}


def get_level(name: str) -> LoadLevel:
    """Return a calibrated load level, or refuse.

    Raises:
        RuntimeError: if §Y.3 calibration has not been run and frozen. Callers
            must not fall back to a default lambda; that would make every
            downstream number a measurement of an arbitrary constant.
    """
    if not CALIBRATED_LEVELS:
        raise RuntimeError(
            "§Y.3 load calibration has not been run. The four lambda values are "
            "not guessable: run scripts/calibrate_load_levels.py, freeze the "
            "resulting table into PREREG_AMENDMENT_2026-07-27_Y, and populate "
            "CALIBRATED_LEVELS from it before any cell fires."
        )
    if name not in CALIBRATED_LEVELS:
        raise KeyError(f"unknown load level {name!r}; have "
                       f"{sorted(CALIBRATED_LEVELS)}")
    return CALIBRATED_LEVELS[name]


# --------------------------------------------------------------------------
# Offered load relative to capacity
# --------------------------------------------------------------------------

def expected_slice_demand(
    substrate: SubstrateNetwork,
    slice_factory: SliceFactory,
    rng: np.random.Generator,
    num_samples: int = 2000,
) -> tuple[float, float]:
    """Mean CPU and RAM demanded by one slice, over `num_samples` draws.

    Sampled from the same factory the episode uses, so the estimate matches the
    workload actually offered rather than a nominal template.
    """
    cpu_total = 0.0
    ram_total = 0.0
    for i in range(num_samples):
        req = slice_factory(
            request_id=f"probe_{i:05d}",
            substrate=substrate,
            rng=rng,
            arrival_time=0.0,
            lifetime=1.0 / SERVICE_RATE,
        )
        cpu_total += sum(v.cpu_demand for v in req.vnfs)
        ram_total += sum(v.ram_demand for v in req.vnfs)
    return cpu_total / num_samples, ram_total / num_samples


def substrate_capacity(substrate: SubstrateNetwork) -> tuple[float, float]:
    """Total CPU and RAM across every node."""
    cpu = sum(substrate.graph.nodes[n]["cpu_capacity"] for n in substrate.graph.nodes)
    ram = sum(substrate.graph.nodes[n]["ram_capacity"] for n in substrate.graph.nodes)
    return float(cpu), float(ram)


def offered_load_fraction(
    substrate: SubstrateNetwork,
    arrival_rate: float,
    expected_cpu: float,
    expected_ram: float,
    service_rate: float = SERVICE_RATE,
) -> float:
    """Offered load as a fraction of substrate capacity.

    A = lambda / mu is the expected number of concurrent slices, so the expected
    resource demanded at any instant is A * E[demand]. Divided by total capacity
    that gives the supply-side load, independent of any approach.

    Returns the binding one of CPU and RAM, not the average: a substrate saturates
    on whichever runs out first, and averaging would hide a bound resource behind
    a free one.
    """
    erlangs = arrival_rate / service_rate
    cpu_cap, ram_cap = substrate_capacity(substrate)
    cpu_frac = (erlangs * expected_cpu / cpu_cap) if cpu_cap > 0 else float("inf")
    ram_frac = (erlangs * expected_ram / ram_cap) if ram_cap > 0 else float("inf")
    return max(cpu_frac, ram_frac)


def arrival_rate_for_rho(
    substrate: SubstrateNetwork,
    rho: float,
    expected_cpu: float,
    expected_ram: float,
    service_rate: float = SERVICE_RATE,
) -> float:
    """Invert `offered_load_fraction`: the lambda that offers load `rho`.

    Lets the §Y.3 sweep be specified once in rho and reused at every size.
    """
    cpu_cap, ram_cap = substrate_capacity(substrate)
    per_erlang = max(expected_cpu / cpu_cap if cpu_cap > 0 else float("inf"),
                     expected_ram / ram_cap if ram_cap > 0 else float("inf"))
    return rho * service_rate / per_erlang


def capacity_by_tier(substrate: SubstrateNetwork) -> dict[str, dict[str, float]]:
    """Per-tier CPU/RAM capacity and node count.

    Aggregate offered load understates difficulty here, because VNF templates are
    tier-restricted: the access and MEC tiers saturate well before the substrate
    as a whole does. §Y.3 records per-tier utilisation alongside the aggregate for
    exactly this reason, and this is the denominator for it.
    """
    out: dict[str, dict[str, float]] = {}
    for node_id in substrate.graph.nodes:
        attrs = substrate.graph.nodes[node_id]
        row = out.setdefault(attrs["tier"], {"nodes": 0.0, "cpu": 0.0, "ram": 0.0})
        row["nodes"] += 1
        row["cpu"] += attrs["cpu_capacity"]
        row["ram"] += attrs["ram_capacity"]
    return out


# --------------------------------------------------------------------------
# Episode construction and the steady-state window
# --------------------------------------------------------------------------

def make_arrival_process(
    substrate: SubstrateNetwork,
    arrival_rate: float,
    rng: np.random.Generator,
    slice_factory: SliceFactory | None = None,
    num_arrivals: int = NUM_ARRIVALS,
) -> ArrivalProcess:
    """Build a §Y episode: Poisson(lambda) arrivals, Exp(mu) lifetimes, mu fixed."""
    ap = ArrivalProcess(
        substrate=substrate,
        num_arrivals=num_arrivals,
        arrival_rate=arrival_rate,
        service_rate=SERVICE_RATE,
        rng=rng,
        slice_factory=slice_factory,
    )
    ap.generate()
    return ap


def steady_state(records: list, warmup: int = WARMUP_ARRIVALS) -> list:
    """Drop the transient prefix from a per-arrival record list.

    `records` must be in arrival order, one entry per ARRIVAL (not per event).
    """
    if len(records) <= warmup:
        raise ValueError(
            f"episode has {len(records)} arrivals, at or below the {warmup}-arrival "
            "warm-up window; nothing would remain to measure"
        )
    return records[warmup:]


def acceptance_ratio(records: list) -> float:
    """Primary §Y metric: admitted / total GENERATED requests.

    Supervisor-ratified definition (2026-07-30): "the number of slice requests
    accepted / total number of generated requests". EVERY arrival in the episode
    counts, including the fill-from-empty transient. No warm-up window.

    `records` entries must be truthy for admitted, falsy for rejected. Admitted
    means committed AND constraint-satisfying: any C1-C9 or QoS-gate violation is
    a rejection. There is no feasibility ceiling and no separate denominator
    (§Y.5).

    This function used to window by WARMUP_ARRIVALS, which made it disagree with
    the number `grid_runner` and `wp7_runner` actually report by up to 3.9 points
    at L4. Use `acceptance_ratio_steady` when the steady-state value is wanted, and
    label it as such wherever it appears.
    """
    if not records:
        return 0.0
    return sum(1 for r in records if r) / len(records)


def acceptance_ratio_steady(records: list, warmup: int = WARMUP_ARRIVALS) -> float:
    """Diagnostic only: acceptance over the steady-state window.

    Not the reported metric. Its distance from `acceptance_ratio` measures how much
    of a run's acceptance comes from arrivals that saw a partly empty substrate,
    which is worth reporting but is not what "acceptance ratio" denotes here.
    """
    window = steady_state(records, warmup)
    return sum(1 for r in window if r) / len(window)
