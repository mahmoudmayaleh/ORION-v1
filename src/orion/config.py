"""ORION configuration system.

All configuration is loaded via Pydantic Settings from environment variables
(prefix ORION_) or .env files. No hardcoded paths — all paths derive from
project_root.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class LLMBackendType(StrEnum):
    CLAUDE = "claude"
    OLLAMA = "ollama"


class TopologyConfig(BaseSettings):
    """Substrate network generation parameters.

    Attributes:
        num_domains: Number of administrative domains.
        nodes_per_domain: Node count per domain (list length must equal num_domains).
        intra_link_density: Erdos-Renyi edge probability for intra-domain links.
        inter_domain_links: Number of inter-domain links between each domain pair.
        tier_distribution: Sampling weights for InfrastructureTier assignment.
        cpu_range: (min, max) vCPUs for node capacity.
        ram_range: (min, max) GB RAM for node capacity.
        bw_range: (min, max) Mbps for link bandwidth capacity.
        delay_intra_range: (min, max) ms propagation delay for intra-domain links.
        delay_inter_range: (min, max) ms propagation delay for inter-domain links.
    """

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


class SlicingConfig(BaseSettings):
    """Slice request generation parameters.

    Attributes:
        sfc_length_range: (min, max) number of VNFs in a service function chain.
        vnf_cpu_range: (min, max) vCPU demand per VNF.
        vnf_ram_range: (min, max) GB RAM demand per VNF.
        flow_bw_range: (min, max) Mbps bandwidth demand per flow edge.
        max_e2e_delay_range: (min, max) ms end-to-end delay budget.
        arrival_rate: Poisson arrival rate lambda (requests/time unit).
        mean_lifetime: Mean slice lifetime in time units (1/mu for Exponential).
    """

    sfc_length_range: tuple[int, int] = (2, 5)
    vnf_cpu_range: tuple[int, int] = (1, 8)
    vnf_ram_range: tuple[int, int] = (2, 32)
    flow_bw_range: tuple[int, int] = (10, 500)
    max_e2e_delay_range: tuple[float, float] = (10.0, 100.0)
    arrival_rate: float = 5.0
    mean_lifetime: float = 10.0

    model_config = {"env_prefix": "ORION_SLICING_"}


class MILPConfig(BaseSettings):
    """MILP oracle solver parameters.

    Attributes:
        solver: PuLP solver identifier string.
        time_limit: Maximum solver wall-clock time in seconds.
        mip_gap: Acceptable MIP optimality gap (relative).
        mu: Admission revenue weight in the objective.
        alpha: Resource cost weight in the objective.
        gamma_intra: Intra-domain bandwidth penalty coefficient.
        gamma_inter: Inter-domain bandwidth penalty coefficient.
        max_inter_domain_hops: Maximum inter-domain hops H for constraint C9.
    """

    solver: str = "PULP_CBC_CMD"
    time_limit: int = 60
    mip_gap: float = 0.01
    mu: float = 100.0
    alpha: float = 1.0
    gamma_intra: float = 0.1
    gamma_inter: float = 1.0
    max_inter_domain_hops: int = 3

    model_config = {"env_prefix": "ORION_MILP_"}


class LLMConfig(BaseSettings):
    """LLM Generator configuration.

    Attributes:
        backend: Active backend type (claude or ollama).
        model_name: Claude model identifier for ClaudeBackend.
        ollama_model: Ollama model tag for OllamaBackend.
        ollama_host: Ollama server base URL.
        k_candidates: Number of candidate plans to generate per request.
        temperature: Sampling temperature for plan generation.
        max_retries: Self-correction retry budget.
        few_shot_count: Number of few-shot examples to include in each prompt.
    """

    backend: LLMBackendType = LLMBackendType.CLAUDE
    model_name: str = "claude-sonnet-4-6"
    ollama_model: str = "mistral:7b"
    ollama_host: str = "http://localhost:11434"
    k_candidates: int = 5
    temperature: float = 0.8
    max_retries: int = 3
    few_shot_count: int = 3

    model_config = {"env_prefix": "ORION_LLM_"}


class RLConfig(BaseSettings):
    """PPO RL Selector training configuration.

    Attributes:
        total_timesteps: Total environment steps for training.
        learning_rate: Adam learning rate.
        gamma: Discount factor.
        clip_range: PPO clipping epsilon.
        n_steps: Steps per rollout buffer.
        batch_size: Mini-batch size for gradient updates.
        n_epochs: Gradient epochs per rollout.
        ent_coef: Entropy bonus coefficient.
        lambda_viol: Constraint violation penalty coefficient in reward.
        eta_milp: MILP proximity bonus weight (applied in 20% of episodes).
        checkpoint_freq: Checkpoint save interval in timesteps.
    """

    total_timesteps: int = 500_000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    clip_range: float = 0.2
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    ent_coef: float = 0.01
    lambda_viol: float = 10.0
    eta_milp: float = 0.5
    checkpoint_freq: int = 50_000

    model_config = {"env_prefix": "ORION_RL_"}


class OrionConfig(BaseSettings):
    """Root configuration for the ORION system.

    All sub-configs are nested and can be overridden via environment variables
    using their respective prefixes or via a .env file.

    Attributes:
        project_root: Absolute path to the repository root.
        topology: Substrate topology generation config.
        slicing: Slice request generation config.
        milp: MILP oracle config.
        llm: LLM Generator config.
        rl: RL Selector training config.
        seed: Global random seed for reproducibility.
    """

    project_root: Path = Path("/home/mmayaleh/Downloads/ORION")
    topology: TopologyConfig = Field(default_factory=TopologyConfig)
    slicing: SlicingConfig = Field(default_factory=SlicingConfig)
    milp: MILPConfig = Field(default_factory=MILPConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    rl: RLConfig = Field(default_factory=RLConfig)
    seed: int = 42

    model_config = {"env_prefix": "ORION_", "env_file": ".env", "env_nested_delimiter": "__"}
