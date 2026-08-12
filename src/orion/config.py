"""ORION configuration."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

# --- MDO observation reference constants (frozen pre-run; PREREG 2026-07-11 §M.4-Δ4) ---
# Largest per-VNF demand in the slice generator (_VNF_TEMPLATES, MediaProc upper bounds
# cpu=(8,16), ram=(16,32)). FROZEN LITERALS BY DESIGN: they pick which node a domain
# reports as its best-fitting one (h^m, see mdo/observation.build_domain_summaries). If the
# VNF templates are later edited these must NOT silently move, so they are not a lookup;
# `test_headroom_refs_still_bound_templates` fails loudly if a template outgrows them.
MDO_HEADROOM_CPU_REF: float = 16.0
MDO_HEADROOM_RAM_REF: float = 32.0

# --- CPU energy estimation constant (PREREG 2026-07-11 §M.4-Δ7) ---
# Measured CPU energy is UNAVAILABLE on the experiment box without root: RAPL sysfs is
# permission-denied, the perf RAPL PMU is blocked (perf_event_paranoid=4), and k10temp exposes
# no power sensor. CPU cost is therefore reported PRIMARILY as measured CPU-seconds; this
# constant yields a clearly-labelled Joules ESTIMATE = cpu_seconds * MDO_CPU_WATT_PER_CORE.
# Value = TDP / cores for the AMD Threadripper PRO 5975WX (280 W / 32 cores). Upper-bound-ish:
# assumes full per-core active power, ignores DVFS/idle. NOT a measured counter. Frozen literal.
MDO_CPU_WATT_PER_CORE: float = 8.75


class TopologyConfig(BaseSettings):
    """Substrate network generation parameters."""

    num_domains: int = 3
    nodes_per_domain: list[int] = Field(default=[8, 10, 12])
    intra_link_density: float = 0.4
    inter_domain_links: int = 4
    tier_distribution: dict[str, float] = Field(
        default={
            "edge": 0.6,
            "regional_cloud": 0.2,
            "central_cloud": 0.2,
        }
    )
    cpu_range: tuple[int, int] = (8, 64)
    ram_range: tuple[int, int] = (16, 256)
    bw_range: tuple[int, int] = (100, 10000)
    delay_intra_range: tuple[float, float] = (0.5, 5.0)
    delay_inter_range: tuple[float, float] = (5.0, 20.0)

    model_config = {"env_prefix": "ORION_TOPOLOGY_"}


class MILPConfig(BaseSettings):
    """MILP solver parameters."""

    solver: str = "PULP_CBC_CMD"
    time_limit: int = 60
    mip_gap: float = 0.01
    mu: float = 100.0
    alpha: float = 1.0
    gamma_intra: float = 0.1
    gamma_inter: float = 1.0
    max_inter_domain_hops: int = 3

    model_config = {"env_prefix": "ORION_MILP_"}


class OrionConfig(BaseSettings):
    """Root configuration."""

    project_root: Path = Path(".")
    topology: TopologyConfig = Field(default_factory=TopologyConfig)
    milp: MILPConfig = Field(default_factory=MILPConfig)
    seed: int = 42

    model_config = {"env_prefix": "ORION_", "env_file": ".env", "env_nested_delimiter": "__"}
