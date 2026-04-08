"""ORION configuration."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class TopologyConfig(BaseSettings):
    """Substrate network generation parameters."""

    num_domains: int = 3
    nodes_per_domain: list[int] = Field(default=[8, 10, 12])
    intra_link_density: float = 0.4
    inter_domain_links: int = 4
    tier_distribution: dict[str, float] = Field(
        default={
            "ran_edge": 0.3,
            "mec": 0.3,
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
