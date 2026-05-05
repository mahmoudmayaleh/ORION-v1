"""Structural constraint checker for Agent B abstract plans.

Validates C4 (resource sufficiency), C5 (inter-domain bandwidth), C8 (tier
placement rules), and schema consistency before a plan reaches the MDO.
Runs in O(|F_s| + |E_s|) time on the abstract topology — no per-node
substrate access needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Violation:
    """A single constraint violation."""

    constraint: str
    vnf_id: str | None
    detail: str


@dataclass
class CheckResult:
    """Output of the structural checker."""

    is_valid: bool
    violations: list[Violation] = field(default_factory=list)

    def summary(self) -> str:
        if self.is_valid:
            return "PASS: Plan is structurally valid."
        lines = [f"FAIL: {len(self.violations)} violation(s):"]
        for v in self.violations:
            tag = f"[{v.constraint}]"
            vnf = f" vnf={v.vnf_id}" if v.vnf_id else ""
            lines.append(f"  {tag}{vnf}: {v.detail}")
        return "\n".join(lines)

    def violation_text_for_prompt(self) -> str:
        """Format violations as feedback for Agent B's retry prompt."""
        if self.is_valid:
            return ""
        parts = []
        for v in self.violations:
            vnf = f" (vnf {v.vnf_id})" if v.vnf_id else ""
            parts.append(f"- {v.constraint}{vnf}: {v.detail}")
        return "\n".join(parts)


def check_plan(
    plan: dict,
    slice_request: dict,
    abstract_topology: dict,
) -> CheckResult:
    """Validate an Agent B abstract plan against C4, C5, C8, and schema.

    Args:
        plan: Agent B output with ``vnf_assignments`` and ``flow_requirements``.
        slice_request: The slice request that produced this plan. Must contain
            ``vnfs`` (each with ``vnf_id``, ``permitted_tiers``) and
            ``flow_edges``.
        abstract_topology: Abstract topology with ``domains`` (each with
            ``domain_id``, ``dominant_tiers``, ``cpu_residual``,
            ``ram_residual``) and ``inter_domain_links``.

    Returns:
        CheckResult indicating pass/fail with violation details.
    """
    violations: list[Violation] = []

    # Build lookup tables
    domain_map = {
        d["domain_id"]: d for d in abstract_topology["domains"]
    }
    vnf_spec_map = {v["vnf_id"]: v for v in slice_request["vnfs"]}
    assignments = plan.get("vnf_assignments", [])
    assignment_map = {a["vnf_id"]: a for a in assignments}

    # ── Schema checks ────────────────────────────────────────────────────

    # Every VNF in the request must appear in the plan
    request_vnf_ids = {v["vnf_id"] for v in slice_request["vnfs"]}
    plan_vnf_ids = {a["vnf_id"] for a in assignments}

    for missing in request_vnf_ids - plan_vnf_ids:
        violations.append(Violation(
            constraint="SCHEMA",
            vnf_id=missing,
            detail="VNF in slice request but missing from plan.",
        ))

    for extra in plan_vnf_ids - request_vnf_ids:
        violations.append(Violation(
            constraint="SCHEMA",
            vnf_id=extra,
            detail="VNF in plan but not in slice request.",
        ))

    # Each VNF must be assigned to exactly one domain
    for a in assignments:
        if a.get("domain") not in domain_map:
            violations.append(Violation(
                constraint="SCHEMA",
                vnf_id=a["vnf_id"],
                detail=f"Assigned to unknown domain '{a.get('domain')}'.",
            ))

    # ── C8: Tier placement rules ─────────────────────────────────────────

    for a in assignments:
        vnf_id = a["vnf_id"]
        domain_id = a.get("domain")
        required_tier = a.get("required_tier")

        spec = vnf_spec_map.get(vnf_id)
        domain = domain_map.get(domain_id)
        if spec is None or domain is None:
            continue  # Already caught by schema checks

        permitted_tiers = spec.get("permitted_tiers", [])

        # The required_tier stated in the plan must be in the VNF's permitted set
        if required_tier and required_tier not in permitted_tiers:
            violations.append(Violation(
                constraint="C8",
                vnf_id=vnf_id,
                detail=(
                    f"required_tier '{required_tier}' not in VNF's "
                    f"permitted_tiers {permitted_tiers}."
                ),
            ))

        # The assigned domain must support the required tier
        domain_tiers = domain.get("dominant_tiers", [])
        if required_tier and required_tier not in domain_tiers:
            violations.append(Violation(
                constraint="C8",
                vnf_id=vnf_id,
                detail=(
                    f"Domain '{domain_id}' tiers {domain_tiers} do not "
                    f"include required_tier '{required_tier}'."
                ),
            ))

        # Even if required_tier is missing, check that at least one
        # permitted tier overlaps with the domain's tiers
        if not required_tier:
            if not set(permitted_tiers) & set(domain_tiers):
                violations.append(Violation(
                    constraint="C8",
                    vnf_id=vnf_id,
                    detail=(
                        f"No tier overlap between VNF permitted_tiers "
                        f"{permitted_tiers} and domain '{domain_id}' "
                        f"tiers {domain_tiers}."
                    ),
                ))

    # ── C4: Aggregate resource sufficiency per domain ────────────────────

    domain_cpu_used: dict[str, float] = {}
    domain_ram_used: dict[str, float] = {}

    for a in assignments:
        domain_id = a.get("domain")
        if domain_id not in domain_map:
            continue

        cpu = a.get("cpu_demand", 0.0)
        ram = a.get("ram_demand", 0.0)
        domain_cpu_used[domain_id] = domain_cpu_used.get(domain_id, 0.0) + cpu
        domain_ram_used[domain_id] = domain_ram_used.get(domain_id, 0.0) + ram

    for domain_id, cpu_used in domain_cpu_used.items():
        domain = domain_map[domain_id]
        cpu_avail = domain["cpu_residual"]
        if cpu_used > cpu_avail:
            violations.append(Violation(
                constraint="C4",
                vnf_id=None,
                detail=(
                    f"Domain '{domain_id}' CPU overcommit: "
                    f"{cpu_used:.1f} demanded > {cpu_avail:.1f} residual."
                ),
            ))

    for domain_id, ram_used in domain_ram_used.items():
        domain = domain_map[domain_id]
        ram_avail = domain["ram_residual"]
        if ram_used > ram_avail:
            violations.append(Violation(
                constraint="C4",
                vnf_id=None,
                detail=(
                    f"Domain '{domain_id}' RAM overcommit: "
                    f"{ram_used:.1f} demanded > {ram_avail:.1f} residual."
                ),
            ))

    # ── C5 / SCHEMA: Inter-domain bandwidth and boundary flag checks ────

    # Build inter-domain link lookup: (src, dst) -> link
    link_map: dict[tuple[str, str], dict] = {}
    for link in abstract_topology.get("inter_domain_links", []):
        src = link.get("source_domain")
        dst = link.get("target_domain")
        if src and dst:
            link_map[(src, dst)] = link

    # Aggregate bandwidth demand per directed inter-domain link
    link_bw_demand: dict[tuple[str, str], float] = {}

    for flow in plan.get("flow_requirements", []):
        src_vnf = flow.get("source_vnf")
        dst_vnf = flow.get("target_vnf")
        bw = flow.get("min_bandwidth_mbps", 0.0)
        crosses = flow.get("crosses_domain_boundary", False)

        src_domain = assignment_map.get(src_vnf, {}).get("domain")
        dst_domain = assignment_map.get(dst_vnf, {}).get("domain")

        if src_domain is None or dst_domain is None:
            continue  # VNFs not found — already caught by schema checks

        actually_crosses = src_domain != dst_domain

        # Boundary flag mismatch
        if actually_crosses and not crosses:
            violations.append(Violation(
                constraint="SCHEMA",
                vnf_id=src_vnf,
                detail=(
                    f"Flow {src_vnf}->{dst_vnf} crosses domains "
                    f"({src_domain}->{dst_domain}) but "
                    f"crosses_domain_boundary=False."
                ),
            ))
        elif not actually_crosses and crosses:
            violations.append(Violation(
                constraint="SCHEMA",
                vnf_id=src_vnf,
                detail=(
                    f"Flow {src_vnf}->{dst_vnf} is intra-domain "
                    f"({src_domain}) but crosses_domain_boundary=True."
                ),
            ))

        # Inter-domain bandwidth checks
        if actually_crosses:
            pair = (src_domain, dst_domain)
            if pair not in link_map:
                violations.append(Violation(
                    constraint="C5",
                    vnf_id=None,
                    detail=(
                        f"No inter-domain link exists between "
                        f"'{src_domain}' and '{dst_domain}' for flow "
                        f"{src_vnf}->{dst_vnf}."
                    ),
                ))
            else:
                link_bw_demand[pair] = link_bw_demand.get(pair, 0.0) + bw

    # Check aggregate bandwidth per link
    for pair, demanded in link_bw_demand.items():
        link = link_map.get(pair)
        if link is None:
            continue  # Already reported as missing
        residual = link.get("bandwidth_residual_mbps", 0.0)
        if demanded > residual:
            violations.append(Violation(
                constraint="C5",
                vnf_id=None,
                detail=(
                    f"Inter-domain link {pair[0]}->{pair[1]} bandwidth "
                    f"overcommit: {demanded:.1f} Mbps demanded > "
                    f"{residual:.1f} Mbps residual."
                ),
            ))

    return CheckResult(
        is_valid=len(violations) == 0,
        violations=violations,
    )
