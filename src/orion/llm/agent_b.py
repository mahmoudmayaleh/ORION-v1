"""Agent B — Abstract Plan Proposer.

Emits a single abstract plan per slice arrival. Operates at the service and
tier level only — never sees raw per-node substrate state.

Inputs (v6 Section 5.2):
  1. Technical slice request s
  2. Abstract topology (per-domain aggregates + inter-domain links)
  3. Semantic reference from K^B (top-k entries) — optional, passed as few-shot
  4. Episodic few-shot from M^B (top-k entries) — optional, passed as few-shot
  5. Feedback context (inference retry only) — violation history h_t

Output:
  Single abstract plan with per-VNF domain assignment (suggested partition),
  resource demands, and per-flow bandwidth requirements.
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from orion.llm.episodic_memory import EpisodicMemory
    from orion.llm.semantic_memory import SemanticMemory

from orion.llm.llm_backend import LLMBackend, extract_json
from orion.llm.structural_checker import CheckResult, check_plan
from orion.profiling import profiled

logger = logging.getLogger("orion.llm.agent_b")


# ── Output schema (grammar-constrained decoding) ─────────────────────────────
# Passed to the LLM endpoint as a json_object *schema*; llama.cpp converts it to
# a GBNF grammar and constrains decoding to exactly this structure. Format drift
# (wrong/missing keys) becomes impossible by construction, so the structural
# checker only ever sees well-formed JSON and validates *semantics*. This mirrors
# the prompt's OUTPUT FORMAT block field-for-field.

class _AssignmentSchema(BaseModel):
    vnf_id: str
    domain: str
    required_tier: str
    cpu_demand: float
    ram_demand: float


class _FlowSchema(BaseModel):
    source_vnf: str
    target_vnf: str
    min_bandwidth_mbps: float
    crosses_domain_boundary: bool


class AgentBPlanSchema(BaseModel):
    plan_id: str
    vnf_assignments: list[_AssignmentSchema]
    flow_requirements: list[_FlowSchema]


AGENT_B_PLAN_JSON_SCHEMA = AgentBPlanSchema.model_json_schema()


def tier_feasible_domains(vnf: dict, abstract_topology: dict) -> list[str]:
    """D(tau_fk): domains whose dominant_tiers overlap the VNF's permitted_tiers (v6 5.2)."""
    permitted = set(vnf.get("permitted_tiers", []))
    out = []
    for d in abstract_topology.get("domains", []):
        if permitted & set(d.get("dominant_tiers", [])):
            out.append(d["domain_id"])
    return out


#: §AC (2026-08-19) -- capacity in the DECODE MASK, not in the prompt.
#:
#: Measured on 40 real L3 arrivals, seed 42, M^B off, cache bypassed, temperature 0:
#: `partial_obs_builder` colocates 100% of chains (1.00 domains) and Agent B splits
#: 85% (2.30 domains, histogram {1:6, 2:20, 3:10, 4:4}), even though a whole-chain
#: host exists on every one of those arrivals. It is not an information problem --
#: the system prompt already carries the ordered decision procedure ("assign EVERY
#: VNF to the single domain with the largest margin, and stop... Only if that list
#: is empty may you split") and the topology already carries
#: `largest_free_node_by_tier`. The model is told to colocate, shown the numbers
#: that say it can, and splits anyway.
#:
#: Restating the derived set as PROSE is known to make it worse: see the note in
#: `_build_user_prompt` where an admissible-domains block raised the split rate
#: 78.4% -> 80.2% and 77.2% -> 96.4% on paired L3 probes. So the correction goes
#: where the tier contract already goes, into the grammar, where the model cannot
#: anchor on it or argue with it.
#:
#: This adds NO information. Every quantity is read out of the two dicts this
#: function is already handed, and the three tests are the ones
#: `_PartitionView.colocation_candidate` applies to the heuristic's own proposal.
#: It only stops the decoder from expressing a partition those inputs already rule
#: out. Off by default so every banked cell reproduces.
CAPACITY_MASK = os.environ.get("ORION_CAPACITY_MASK", "0") != "0"

#: §AD (2026-08-19) -- colocation as the plan's SHAPE, not as a preference.
#:
#: The decode mask (§AC) narrows WHICH domains are selectable but leaves the shape
#: free: with two qualifying hosts the model can still put VNF 1 in one and VNF 2 in
#: the other. Measured L3 seed 42, splitting is what is left of the gap once the h^m
#: guard is applied -- `cross_domain_infeasible` + `c9_hops` run 447 per 2000 for the
#: guarded LLM plan against 9 for `MDO-partial`, which is 22.4% of arrivals, while the
#: capacity bin has already gone the other way (16 against 98).
#:
#: So this changes the CONTRACT. When some domain can host the whole chain the model
#: emits ONE `host_domain` chosen from those domains and nothing else; `expand_host_domain`
#: turns that into the per-VNF assignments every downstream stage already expects. A
#: split is then not discouraged, it is unrepresentable. When no host exists the schema
#: falls back to per-VNF assignments, which is exactly when a split is the right answer
#: and the model's judgement about WHERE to cut is worth having.
#:
#: This is `partial_obs_builder`'s step 3 / step 4 moved out of the prompt and into the
#: grammar. It leaves Agent B real authorship -- which host, and how to cut the hard
#: cases -- and it costs 1 decoded field instead of K.
COLOCATION_CONTRACT = os.environ.get("ORION_COLOCATION_CONTRACT", "0") != "0"

#: §AD.1 -- keep only hosts whose slack is within this fraction of the best host's.
#:
#: Measured L3 seed 42 under the contract: Agent B chose domain `d0` on 30/30
#: arrivals, and reversing the enum did not move it (same DOMAIN 30/30, same INDEX
#: 0/30), so this is a fixed preference over domains and not enum position. Every
#: slice therefore lands in one domain, which drives it to its margin while it still
#: passes the tests -- and `fits_a_node` is NECESSARY, NOT SUFFICIENT (it reports the
#: best node per tier, so it cannot see two co-located functions competing for the
#: same node). That blind spot bites precisely at the margin, which is why the
#: contract eliminated every split (cross 215->0, c9 78->0) and still tripled the
#: actor bin (299 -> 704).
#:
#: `partial_obs_builder` never reaches that regime because it always takes the
#: LARGEST slack, which load-balances as domains fill. This band gives the model the
#: same protection without taking the choice away: it may pick any host that is still
#: competitive, but it can no longer ride one domain down while a much emptier one
#: sits in the enum. 0.0 keeps every qualifying host (shipped behaviour).
HOST_SLACK_BAND = float(os.environ.get("ORION_HOST_SLACK_BAND", "0") or 0.0)


def expand_host_domain(plan: dict, slice_request: dict) -> bool:
    """Expand {"host_domain": d} into the per-VNF assignment list, in place.

    Runs immediately after parsing and BEFORE `fill_derived_fields`, so the defensive
    validator, `recompute_required_tiers`, `recompute_flow_boundaries`, `check_plan`
    and the `PlanSummary` build all see the shape they have always seen. Returns True
    if it expanded, False if the plan was already in per-VNF form.
    """
    host = plan.pop("host_domain", None)
    if host is None:
        return False
    plan["vnf_assignments"] = [
        {"vnf_id": v.get("vnf_id"), "domain": host}
        for v in (slice_request.get("vnfs") or [])
    ]
    return True


def whole_chain_hosts(slice_request: dict, abstract_topology: dict) -> list[str]:
    """Domains that can host the ENTIRE chain, best aggregate slack first.

    Three tests, all on the observation surface:
      (a) every VNF is tier-feasible in the domain,
      (b) for every VNF some permitted tier's `largest_free_node_by_tier` seats it
          on ONE node (h^m: a tier can hold 300 free CPU and still not seat a VNF
          needing 12 if its biggest free node has 8), and
      (c) the domain's aggregate residual covers the whole chain with slack > 0.

    Returns [] when no single domain can take the chain, which is exactly when a
    split is legitimate.
    """
    vnfs = slice_request.get("vnfs", [])
    if not vnfs:
        return []
    tot_cpu = sum(float(v.get("cpu_demand") or 0.0) for v in vnfs)
    tot_ram = sum(float(v.get("ram_demand") or 0.0) for v in vnfs)
    scored: list[tuple[float, str]] = []
    for d in abstract_topology.get("domains", []):
        dom_tiers = set(d.get("dominant_tiers", []))
        h = d.get("largest_free_node_by_tier", {}) or {}
        for v in vnfs:
            permitted = set(v.get("permitted_tiers", [])) & dom_tiers
            if not permitted:
                break
            cpu_d = float(v.get("cpu_demand") or 0.0)
            ram_d = float(v.get("ram_demand") or 0.0)
            if not any(float((h.get(t) or {}).get("cpu") or 0.0) >= cpu_d
                       and float((h.get(t) or {}).get("ram") or 0.0) >= ram_d
                       for t in permitted):
                break
        else:
            slack = min(float(d.get("cpu_residual") or 0.0) - tot_cpu,
                        float(d.get("ram_residual") or 0.0) - tot_ram)
            if slack > 0:
                scored.append((slack, d["domain_id"]))
    scored.sort(key=lambda x: (-x[0], x[1]))
    if HOST_SLACK_BAND > 0.0 and scored:
        cutoff = scored[0][0] * HOST_SLACK_BAND
        scored = [x for x in scored if x[0] >= cutoff]
    return [dom for _, dom in scored]


#: Completion ceiling for a plan call. The server reserves this on TOP of the
#: prompt, so leaving it at the 2048 default cost both overnight jobs: a 4121-token
#: prompt plus a 2048-token reservation is 6169 against a 4096 window, and the
#: request is refused with a 400 rather than truncated. Measured plans are 50 to 90
#: tokens since the schema asks for the partition only, so this is ~4x headroom and
#: still leaves a plan-carrying request inside a 4608-token window.
PLAN_MAX_TOKENS = 384


def build_pinned_plan_schema(slice_request: dict, abstract_topology: dict) -> dict | None:
    """Per-request JSON schema that PINS the interface contract, not just shape.

    Beyond the shape the static AGENT_B_PLAN_JSON_SCHEMA fixes, this constrains decoding to the
    v6 5.2 interface: each suggested domain m~(f_k) must lie in D(tau_fk), the tier-feasible set,
    identical to the MDO action-space mask (5 3.4). It uses PER-POSITION (tuple) assignment
    schemas (server honors both draft-07 items-list and 2020-12 prefixItems):
      - position i is pinned to VNF i's id via a singleton enum (exact bijection, no omit/dup),
      - domain at position i is an enum of ONLY that VNF's tier-feasible domains D(tau_fk),
      - required_tier is a permissive enum (recomputed deterministically post-generation).
    Grammar-valid output can no longer name a nonexistent VNF, an invalid domain, OR a
    tier-infeasible domain (a contract violation the RL approaches structurally cannot make). Only
    genuine C4 (resource) / C5 (inter-domain bandwidth/reachability) infeasibility can remain --
    exactly the prior-quality signal the model owns.

    Returns None if any VNF has NO tier-feasible domain: the slice is genuinely unplaceable, so
    the caller should structural-reject WITHOUT an LLM call (no schema can rescue it).
    """
    vnfs = slice_request.get("vnfs", [])
    vnf_ids = [v["vnf_id"] for v in vnfs]
    # §AC: when some domain can take the whole chain, restrict EVERY position to
    # those domains. The model still chooses which one, and may still split, but
    # only across domains that each individually pass the same admissibility tests
    # the heuristic applies -- so a split can no longer name a domain that cannot
    # seat the function it is given. When the list is empty a split is legitimate
    # and the mask falls back to the tier contract, unchanged.
    # The contract IMPLIES the mask: it is defined in terms of the whole-chain hosts,
    # so gating it on CAPACITY_MASK alone would let ORION_COLOCATION_CONTRACT=1 be set
    # on its own and silently do nothing.
    hosts = (whole_chain_hosts(slice_request, abstract_topology)
             if (CAPACITY_MASK or COLOCATION_CONTRACT) else [])

    def assignment_for(v):
        feas = tier_feasible_domains(v, abstract_topology)
        if not feas:
            return None  # signals genuine tier-infeasibility for this VNF
        if hosts:
            feas = list(hosts)
        tiers = sorted(v.get("permitted_tiers", []))
        # required_tier, cpu_demand and ram_demand are NOT asked for. The first is
        # overwritten by recompute_required_tiers and the other two are echoes of the
        # request, so generating them cost ~200 of the ~257 output tokens per call and
        # bought nothing. Measured 2026-08-07: 9.3 s per call at ~28 tok/s effective,
        # which put the LLM grid at ~350 h. `tiers` stays in scope for the caller.
        _ = tiers
        return {
            "type": "object",
            "properties": {
                "vnf_id": {"enum": [v["vnf_id"]]},          # pin position -> VNF
                "domain": {"enum": feas},                    # D(tau_fk) only
            },
            "required": ["vnf_id", "domain"],
            "additionalProperties": False,
        }

    if COLOCATION_CONTRACT and hosts:
        # A non-empty `hosts` already implies every VNF is tier-feasible in each of
        # them, so the None contract below cannot be reachable from here.
        return {
            "type": "object",
            "properties": {"host_domain": {"enum": list(hosts)}},
            "required": ["host_domain"],
            "additionalProperties": False,
        }

    per_position = []
    for v in vnfs:
        a = assignment_for(v)
        if a is None:
            return None  # genuinely infeasible slice
        per_position.append(a)

    # flow_requirements is filled from the request (see fill_derived_fields), so the
    # structural check now screens on the TRUE per-edge bandwidth instead of a number the
    # model invented. plan_id is read by nothing.
    k = len(vnf_ids)
    return {
        "type": "object",
        "properties": {
            # per-position tuple validation (draft-07 items-list): exactly K, position i == VNF i
            "vnf_assignments": {"type": "array", "minItems": k, "maxItems": k, "items": per_position},
        },
        "required": ["vnf_assignments"],
        "additionalProperties": False,
    }


def recompute_required_tiers(plan: dict, slice_request: dict, abstract_topology: dict) -> None:
    """Set each assignment's required_tier to a tier in (permitted ∩ chosen-domain tiers), in place.

    required_tier must be BOTH in the VNF's permitted set and supported by the assigned domain
    (C8). With domain pinned to a tier-feasible one, that intersection is non-empty; the model's
    stated required_tier can still mismatch the specific chosen domain, so it is recomputed like
    crosses_domain_boundary -- a derived field, not a placement decision.
    """
    permitted = {v["vnf_id"]: set(v.get("permitted_tiers", []))
                 for v in slice_request.get("vnfs", [])}
    dom_tiers = {d["domain_id"]: set(d.get("dominant_tiers", []))
                 for d in abstract_topology.get("domains", [])}
    for a in plan.get("vnf_assignments", []) or []:
        vid, dom = a.get("vnf_id"), a.get("domain")
        inter = permitted.get(vid, set()) & dom_tiers.get(dom, set())
        if inter:
            a["required_tier"] = sorted(inter)[0]


def fill_derived_fields(plan: dict, slice_request: dict) -> None:
    """Fill every field the model no longer emits, in place, from the REQUEST.

    The model emits the partition only. The rest is either a copy of the request
    (cpu_demand, ram_demand, min_bandwidth_mbps) or derived from the assignments
    (crosses_domain_boundary, set by recompute_flow_boundaries; required_tier, by
    recompute_required_tiers), so generating it cost decode tokens and bought
    nothing. Taking min_bandwidth_mbps from the request also means the structural
    check screens on the real per-edge demand and not on an invented number.

    This runs immediately after parsing so the plan the defensive validator sees
    has the shape it has always had. Fill only; never overwrite.
    """
    demands = {v.get("vnf_id"): v for v in slice_request.get("vnfs", []) or []}
    plan.setdefault("plan_id", str(slice_request.get("request_id", "req")) + "_plan")
    for a in plan.get("vnf_assignments", []) or []:
        v = demands.get(a.get("vnf_id"), {})
        a.setdefault("cpu_demand", v.get("cpu_demand", 0.0))
        a.setdefault("ram_demand", v.get("ram_demand", 0.0))
        a.setdefault("required_tier", "")
    if plan.get("flow_requirements"):
        return
    plan["flow_requirements"] = [
        {
            "source_vnf": f.get("source_vnf"),
            "target_vnf": f.get("target_vnf"),
            "min_bandwidth_mbps": f.get("bandwidth_demand", 0.0),
            "crosses_domain_boundary": False,
        }
        for f in slice_request.get("flow_edges", []) or []
    ]


def recompute_flow_boundaries(plan: dict) -> None:
    """Set each flow's crosses_domain_boundary from the plan's OWN assignments, in place.

    crosses_domain_boundary is 100% derived from the VNF->domain assignments, yet the LLM emits
    it as a free field and gets it inconsistent with its own placement on longer chains
    (validity probe 2026-07-10: the dominant post-enum failure — grammar-valid, correct
    partition, rejected only over this computable boolean). Recomputing it does NOT touch the
    LLM's partition decision; it fixes a field the model should never have owned. Genuine
    inter-domain feasibility (C5 reachability/bandwidth, C8 tier) is untouched and still checked.
    """
    dom = {a.get("vnf_id"): a.get("domain") for a in plan.get("vnf_assignments", [])}
    for f in plan.get("flow_requirements", []) or []:
        sd, td = dom.get(f.get("source_vnf")), dom.get(f.get("target_vnf"))
        if sd is not None and td is not None:
            f["crosses_domain_boundary"] = (sd != td)


class PlanTruncationError(ValueError):
    """Raised when the completion hit the context window (finish_reason=length)
    rather than closing naturally. Subclasses ValueError so existing parse-error
    handling still catches it, but lets callers count truncation distinctly from
    a genuine malformed-JSON parse failure (Amendment 3)."""


# ── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are Agent B in a 6G network slice orchestration system. You produce an \
abstract placement plan: an assignment of every VNF in a slice request to one \
network domain. You do not choose nodes. Each domain places its own VNFs on \
its own nodes after you decide.

THE SUBSTRATE
There are 5 administrative domains and exactly 3 infrastructure tiers: edge, \
regional_cloud and central_cloud. No other tier exists.
- Two domains provide all three tiers.
- Two provide edge and regional_cloud.
- One provides central_cloud only. It can host no complete chain, because \
every slice type contains at least one function that cannot run on \
central_cloud. It can only ever receive part of a split chain.
So four of the five domains provide both edge and regional_cloud, and every \
slice type in this workload is tier-feasible in full inside any one of them. \
Putting a whole chain in one domain is therefore almost always possible. \
Splitting is the exception, not the norm.

WHAT YOU RECEIVE
1. A slice request: the chain in order, per-VNF cpu_demand and ram_demand, \
the QoS budget, and each VNF's permitted_tiers.
2. An abstract topology, described in detail below.
3. Reference Knowledge: operator guidelines for this infrastructure. Follow \
these closely.
4. Sometimes Learned Rules, and Past Outcomes from earlier episodes. A past \
outcome gives a plan SHAPE and the network condition it held under. It never \
names domains, because conditions have changed since. Do not try to \
reconstruct which domains were used; choose domains for the network as it is \
now.

WHAT THE NUMBERS MEAN
Each domain reports, for the domain as a whole and again for each of its tiers:
- cpu_residual / ram_residual: free CPU and RAM, SUMMED over all its nodes.
- cpu_capacity / ram_capacity: the same sums when the domain is empty. \
Residual divided by capacity is how loaded that domain or tier is. A large \
residual in a large domain is not the same as a large residual in a small one.
- largest_free_node_by_tier: the single biggest free node in that tier, given \
as its own free cpu and ram. This is the number that decides whether ONE VNF \
fits, because a VNF runs on one node and cannot be spread across several. A \
tier can hold 300 free CPU in total and still not seat a VNF needing 12 if \
its largest free node has 8. Check a VNF's cpu_demand and ram_demand against \
this, not against the tier total.
So: the totals tell you whether a domain can hold the WHOLE chain, and \
largest_free_node_by_tier tells you whether it can hold each INDIVIDUAL VNF. \
Both must hold. Everything here is the live state at this moment, already net \
of every slice currently running.
The domain you emit is already restricted to tier-feasible choices, so you \
cannot violate the tier rule C8 and need not spend effort on it. Your real \
decision is which feasible domains to use, and above all HOW MANY.

DECISION PROCEDURE, in order:
1. Add up cpu_demand over every VNF in the chain, and ram_demand likewise.
2. List the domains that are tier-feasible for EVERY VNF and where BOTH:
   (a) cpu_residual and ram_residual exceed those two totals with margin, and
   (b) for each VNF, some tier of that domain the VNF may use has a \
largest_free_node big enough for that VNF on its own.
3. If that list is not empty, assign EVERY VNF to the single domain on it \
with the largest margin, and stop. This is the expected answer for most \
requests.
4. Only if that list is empty may you split. Then: use the fewest domains \
that work, normally two; put each VNF only where test (b) passes for it; cut \
the chain between consecutive functions so it enters a domain once and never \
returns to one it has left; prefer the cut whose flow has the lowest \
bandwidth demand; and check that flow against bandwidth_residual_mbps on the \
inter-domain link.

WHY SPLITTING IS COSTLY
Every crossing adds propagation and queueing delay against the end-to-end \
budget (C7), consumes inter-domain bandwidth (C5), and counts against a hard \
limit of 3 inter-domain hops for the whole slice (C9). A plan using K \
domains costs at least K-1 crossings, and more if the chain revisits a \
domain. Giving each VNF its own domain is the most expensive plan available \
and is almost never the right answer.

OUTPUT FORMAT — respond ONLY with a single JSON object:
{
  "vnf_assignments": [
    {"vnf_id": "<vnf_id>", "domain": "<domain_id>"}
  ]
}
List every VNF exactly once, in chain order. Output only the JSON, no prose."""


# ── M^B exemplar rendering ───────────────────────────────────────────────────

# An M^B exemplar states the plan SHAPE and the network condition it held under,
# not the concrete domain assignment.
#
# Why: the concrete form makes the model copy. Paired copy test 2026-08-05, 149
# matched decisions at identical substrate state -- exemplars rewrote 33% of
# Agent B's plans and moved feasibility by +0.0 pp (3 better, 3 worse, McNemar
# p=1.0). Every URLLC K=2 went to [d0,d0] and every eMBB K=3 to [d3,d4,d3]
# whatever the congestion, so the same retrieved plan was infeasible at arrival
# 10 and feasible at arrival 18. The exemplar reproduced a plan shape without
# the conditions that made it work.
#
# Two documented failure modes meet here. Case-based reasoning is Retrieve ->
# Reuse -> Revise -> Retain (arXiv:2504.06943); M^B had no Revise step, so a
# retrieved case was applied unadapted. And LLMs exhibit copy bias, echoing
# demonstration answers instead of the pattern, worst when the demonstration
# context resembles the query context (arXiv:2410.01288) -- which under a fixed
# §Y substrate it always does. Trajectory-level retrieval returning plausible
# cases that ignore state-transition dynamics is the same diagnosis TRAD reaches
# (arXiv:2403.06221).
#
# Set False to reproduce the pre-2026-08-05 concrete form.
MB_ABSTRACT_EXEMPLARS = True

_TIER_ORDER = ("edge", "regional_cloud", "central_cloud")


def _condition_line(condition: dict | None) -> str | None:
    """Congestion in the fields that actually move; see condition_signature."""
    if not condition:
        return None
    tiers = condition.get("tier_cpu_residual") or {}
    shown = ", ".join(f"{t} {float(tiers[t]):.2f}" for t in _TIER_ORDER if t in tiers)
    return (f"overall cpu headroom {float(condition.get('cpu_residual_frac', 0.0)):.2f}"
            f" ({condition.get('bucket', 'unknown')})"
            + (f"; per-tier headroom {shown}" if shown else ""))


def _slice_line(slice_request: dict) -> str:
    vnfs = slice_request.get("vnfs", [])
    qos = slice_request.get("qos") or {}
    tiers = sorted({t for v in vnfs for t in (v.get("permitted_tiers") or [])})
    return (f"{slice_request.get('slice_type', 'unknown')}, chain of {len(vnfs)}, "
            f"max delay {qos.get('max_e2e_delay', '?')} ms, "
            f"permitted tiers {tiers}")


def _abstract_exemplar(ex: dict, current_condition: dict | None) -> list[str]:
    """One exemplar as shape + the condition it held under + the condition now.

    No domain identifier appears anywhere in the block. The model is given what
    worked and how the network differs today, and has to choose domains itself.
    """
    out = [f"Past slice: {_slice_line(ex.get('slice_request') or {})}"]

    shape = ex.get("plan_shape") or {}
    strategy = shape.get("strategy")
    tiers = shape.get("tier_assignment") or []
    n_dom = len(shape.get("domains_used") or [])
    if not strategy:
        # Counter-examples and pre-§Y entries carry no plan_shape. Derive width
        # and tiers from the stored assignment rather than dropping the case:
        # an exemplar with no placement information is not worth a prompt slot.
        assigns = (ex.get("placement_plan") or {}).get("vnf_assignments") or []
        doms = {a.get("domain") for a in assigns if a.get("domain")}
        if doms:
            n_dom = len(doms)
            strategy = "co-locate" if n_dom <= 1 else "split"
            tiers = [a.get("required_tier") for a in assigns if a.get("required_tier")]
        elif ex.get("committed_partition") is not None:
            n_dom = len(set(ex["committed_partition"]))
            strategy = "co-locate" if n_dom <= 1 else "split"

    if strategy:
        placed = f"{strategy} across {n_dom} domain(s)" if n_dom else str(strategy)
        out.append(f"Plan shape that was tried: {placed}"
                   + (f", tiers {tiers}" if tiers else ""))
        cuts = shape.get("cut_points") or []
        if cuts:
            out.append(f"Chain was cut between: {cuts}")

    when = _condition_line(ex.get("condition"))
    if when:
        out.append(f"Network then: {when}")
    now = _condition_line(current_condition)
    if now:
        out.append(f"Network now:  {now}")

    if ex.get("outcome"):
        out.append(f"Outcome: {ex['outcome']}.")
    return out


# ── Prompt builder ───────────────────────────────────────────────────────────

def build_user_prompt(
    slice_request: dict,
    abstract_topology: dict,
    few_shot_examples: list[dict] | None = None,
    violation_feedback: str | None = None,
    reference_knowledge: str | None = None,
    current_condition: dict | None = None,
    insights: str | None = None,
) -> str:
    """Build the user message for Agent B.

    Args:
        slice_request: The slice request dict.
        abstract_topology: Abstract topology dict from build_abstract_topology().
        few_shot_examples: Optional list of {slice_request, placement_plan} dicts
            from episodic memory M^B.
        violation_feedback: Optional violation text from a failed structural check,
            appended on retry/replan rounds.
        reference_knowledge: Optional formatted text from K^B semantic memory,
            injected as a Reference Knowledge block above the few-shot examples.
        current_condition: Condition signature at decision time. Rendered beside
            each exemplar's own condition so the difference is explicit rather
            than left for the model to infer.
        insights: Formatted rule block from insight_extraction.format_insights.
            ExpeL concatenates insights into the task specification alongside
            the retrieved cases, so this sits above them and below K^B.
    """
    parts: list[str] = []

    # Reference knowledge from K^B (above few-shot, per v6 Section 4.5)
    if reference_knowledge:
        parts.append(reference_knowledge)

    # Distilled rules from M^B (ExpeL insight extraction)
    if insights:
        parts.append("\n" + insights)

    # Few-shot examples from M^B
    if few_shot_examples:
        if MB_ABSTRACT_EXEMPLARS:
            parts.append(
                "\n--- Past Outcomes (shapes, not assignments) ---"
                "\nThese record what plan SHAPE worked under what network"
                " condition. The domains are not given: conditions have changed,"
                " so choose domains for the network as it is now.")
            for i, ex in enumerate(few_shot_examples, 1):
                parts.append(f"\n--- Case {i} ---")
                parts.extend(_abstract_exemplar(ex, current_condition))
        else:
            parts.append("\n--- Past Plans (Few-Shot Examples) ---")
            for i, ex in enumerate(few_shot_examples, 1):
                parts.append(f"\n--- Example {i} ---")
                parts.append(f"Slice Request:\n{json.dumps(ex['slice_request'], indent=2)}")
                parts.append(
                    f"You suggested this plan:\n{json.dumps(ex['placement_plan'], indent=2)}")
                # Second outcome loop: what the RL coordinator actually committed + verdict,
                # so the next plan is steered toward what gets committed and verified.
                if ex.get("committed_partition") is not None:
                    dv = " (DIVERGED from your suggestion)" if ex.get("diverged") else ""
                    parts.append(
                        f"The RL coordinator committed partition {ex['committed_partition']}{dv}.")
                if ex.get("outcome"):
                    parts.append(f"Ground-truth verdict: {ex['outcome']}.")

    # Current task
    parts.append("\n--- Current Task ---")
    parts.append(f"\nAbstract Topology:\n{json.dumps(abstract_topology, indent=2)}")
    parts.append(f"\nSlice Request:\n{json.dumps(slice_request, indent=2)}")

    # There is deliberately no derived "admissible domains" block here. It was
    # measured twice on paired L3 probes over 167 identical states and raised the
    # split rate both times, 78.4% -> 80.2% under K^B v1.0 and 77.2% -> 96.4% under
    # v2.0, with co-located plans falling from 38 of 167 to 6 and the first-VNF
    # choice collapsing onto two domains. Restating the feasible set as prose
    # anchored the model on the list instead of informing it. The admissible set
    # still binds, as the decode mask in `build_pinned_plan_schema`.

    # Violation feedback for retry
    if violation_feedback:
        parts.append(
            f"\n--- PREVIOUS ATTEMPT FAILED ---\n"
            f"Your previous plan violated the following constraints. "
            f"Fix these issues in your new plan:\n{violation_feedback}"
        )

    return "\n".join(parts)


# ── Agent B ──────────────────────────────────────────────────────────────────

class AgentB:
    """Abstract plan proposer.

    Generates a single plan per invocation. Supports structural-check retry
    (one regeneration attempt with violation feedback).

    Args:
        llm: LLM backend for chat completion.
        system_prompt: Override the default system prompt if needed.
    """

    def __init__(
        self,
        llm: LLMBackend,
        system_prompt: str | None = None,
    ) -> None:
        self.llm = llm
        self.system_prompt = system_prompt or SYSTEM_PROMPT

    def generate_plan(
        self,
        slice_request: dict,
        abstract_topology: dict,
        few_shot_examples: list[dict] | None = None,
        violation_feedback: str | None = None,
        reference_knowledge: str | None = None,
        plan_schema: dict | None = None,
        current_condition: dict | None = None,
        insights: str | None = None,
    ) -> dict:
        """Generate one abstract plan via LLM call.

        Args:
            slice_request: Slice request dict.
            abstract_topology: Abstract topology dict.
            few_shot_examples: Few-shot examples from M^B episodic memory.
            violation_feedback: Violation text from a prior failed attempt.
            reference_knowledge: Formatted text from K^B semantic memory.
            current_condition: Condition signature at decision time, rendered
                beside each exemplar's own condition.

        Returns:
            Parsed plan dict.

        Raises:
            ValueError: If the LLM output cannot be parsed as JSON.
        """
        user_msg = build_user_prompt(
            slice_request, abstract_topology,
            few_shot_examples, violation_feedback, reference_knowledge,
            current_condition=current_condition, insights=insights,
        )
        raw = self.llm.complete(
            self.system_prompt, user_msg,
            max_tokens=PLAN_MAX_TOKENS,
            response_format={"type": "json_object",
                             "schema": plan_schema or AGENT_B_PLAN_JSON_SCHEMA},
        )
        # Distinguish context-window truncation from a genuine parse failure:
        # a truncated grammar-constrained completion is cut-off JSON, but the
        # cause (prompt too large for n_ctx) and the fix are entirely different.
        if getattr(self.llm, "last_finish_reason", None) == "length":
            raise PlanTruncationError(
                "Agent B completion truncated at n_ctx "
                f"(prompt_tokens={getattr(self.llm, 'last_prompt_tokens', None)}, "
                f"n_ctx={getattr(self.llm.config, 'n_ctx', None)})"
            )
        plan = extract_json(raw)
        expand_host_domain(plan, slice_request)
        fill_derived_fields(plan, slice_request)
        # Defensive secondary check (v6.5): with grammar-constrained decoding this
        # must already hold. If it ever fires, the constrained path has regressed —
        # log loudly rather than silently repair the model's output.
        try:
            AgentBPlanSchema.model_validate(plan)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "agent_b_schema_defensive_fired",
                extra={"error": str(exc)[:200]},
            )
        return plan

    def generate_and_check(
        self,
        slice_request: dict,
        abstract_topology: dict,
        few_shot_examples: list[dict] | None = None,
        max_retries: int = 1,
        reference_knowledge: str | None = None,
        plan_schema: dict | None = None,
        current_condition: dict | None = None,
        insights: str | None = None,
    ) -> tuple[dict, CheckResult]:
        """Generate a plan and validate it, retrying once on structural failure.

        This implements the structural-checker regeneration logic from v6
        Section 5.3: at most one regeneration during training, with violation
        feedback appended to the prompt.

        Args:
            slice_request: Slice request dict.
            abstract_topology: Abstract topology dict.
            few_shot_examples: Few-shot examples from M^B episodic memory.
            max_retries: Number of regeneration attempts on structural failure.
            reference_knowledge: Formatted text from K^B semantic memory.

        Returns:
            Tuple of (plan_dict, CheckResult). The plan may still be invalid
            if all retries exhausted — the caller must check result.is_valid.
        """
        violation_feedback = None
        # Fence: grammar-constrained decoding makes a JSON parse failure
        # unreachable, but initialise so the parse-failure branch can never
        # return an unbound `plan` (the former UnboundLocalError at max_retries=0).
        plan: dict = {}

        for attempt in range(1 + max_retries):
            try:
                with profiled("llm.generate", {"attempt": attempt + 1}):
                    plan = self.generate_plan(
                        slice_request, abstract_topology,
                        few_shot_examples, violation_feedback, reference_knowledge,
                        plan_schema=plan_schema,
                        current_condition=current_condition,
                        insights=insights,
                    )
            except PlanTruncationError as exc:
                # Context-window exhaustion, NOT malformed output. Counted
                # distinctly so an approach-asymmetric truncation regression is
                # visible in results rather than hiding as a parse failure.
                logger.warning(
                    "agent_b_completion_truncated",
                    extra={"attempt": attempt + 1, "detail": str(exc)[:200]},
                )
                result = CheckResult(is_valid=False, violations=[])
                violation_feedback = "Your previous response was cut off. Respond with a single, complete, minimal JSON object."
                continue
            except ValueError:
                logger.warning(
                    "agent_b_json_parse_failed",
                    extra={"attempt": attempt + 1},
                )
                # Treat parse failure as a structural failure
                result = CheckResult(
                    is_valid=False,
                    violations=[],
                )
                violation_feedback = "Your previous response was not valid JSON. Respond with only a JSON object."
                continue

            # Fix the derived fields (crosses_domain_boundary flag, required_tier) before
            # validating, so a valid partition is not rejected over values the LLM should not own.
            recompute_flow_boundaries(plan)
            recompute_required_tiers(plan, slice_request, abstract_topology)
            with profiled("struct.check", {"attempt": attempt + 1}):
                result = check_plan(plan, slice_request, abstract_topology)

            if result.is_valid:
                logger.debug(
                    "agent_b_plan_valid",
                    extra={"attempt": attempt + 1},
                )
                return plan, result

            logger.info(
                "agent_b_structural_check_failed",
                extra={
                    "attempt": attempt + 1,
                    "violations": len(result.violations),
                },
            )
            violation_feedback = result.violation_text_for_prompt()

        return plan, result

    def generate_with_memory(
        self,
        slice_request: dict,
        abstract_topology: dict,
        kb: SemanticMemory | None = None,
        mb: EpisodicMemory | None = None,
        max_retries: int = 1,
        plan_schema: dict | None = None,
        topology_signature: dict | None = None,
        condition_signature: dict | None = None,
        insights: str | None = None,
    ) -> tuple[dict, CheckResult]:
        """Generate a plan using K^B semantic and M^B episodic memory.

        Retrieves relevant context from both memory systems and feeds it
        into generate_and_check().

        Args:
            slice_request: Slice request dict.
            abstract_topology: Abstract topology dict.
            kb: Optional semantic memory (K^B) for reference knowledge.
            mb: Optional episodic memory (M^B) for few-shot examples.
            max_retries: Number of regeneration attempts on structural failure.

        Returns:
            Tuple of (plan_dict, CheckResult).
        """
        from orion.llm.semantic_memory import build_query_from_slice

        query = build_query_from_slice(slice_request, topology=topology_signature,
                                       condition=condition_signature)

        reference_knowledge: str | None = None
        if kb is not None:
            kb_entries = kb.retrieve(query, top_k=5)
            formatted = kb.format_for_prompt(kb_entries)
            if formatted:
                reference_knowledge = formatted

        few_shot_examples: list[dict] | None = None
        if mb is not None:
            # §Y.6: the state term is network CONDITION when one is supplied.
            # topology_signature is the pre-§Y key and is a constant (hence a
            # no-op) on a fixed substrate.
            mb_entries = mb.retrieve(query, top_k=3, topology=topology_signature,
                                     condition=condition_signature)
            converted = mb.to_few_shot(mb_entries)
            if converted:
                few_shot_examples = converted

        return self.generate_and_check(
            slice_request,
            abstract_topology,
            few_shot_examples=few_shot_examples,
            max_retries=max_retries,
            reference_knowledge=reference_knowledge,
            plan_schema=plan_schema,
            current_condition=condition_signature,
            insights=insights,
        )
