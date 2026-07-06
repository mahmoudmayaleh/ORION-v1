"""Autoregressive domain placement policy using pointer-network scoring.

Implements the interleaved place-then-route paradigm:
  For each VNF k in SFC order within the fragment:
    1. Encode graph state via GNN backbone (full re-encode every step)
    2. Encode VNF context via MLP
    3. Score nodes via single-head dot-product pointer (Vinyals et al. 2015,
       Kool et al. 2018 attention variant for combinatorial optimization)
    4. Apply action mask, softmax, sample
    5. Update node residuals (autoregressive state update)
    6. Route flow from previous VNF (if intra-domain), update edge BW

Edge features inform the GATv2 encoder, producing edge-aware node embeddings.
The pointer head scores nodes via single-head dot-product attention with
a VNF query vector -- the scoring step itself does not use edge info directly.

Full re-encoding every step is the default. On domain subgraphs of 8-30 nodes
with 3 GATv2 layers, this is <1ms per forward pass on GPU. Cached-embedding
with delta fusion (PARCO-style) is available as an optimization if profiling
shows re-encoding is a bottleneck during training.

The policy includes a learned NULL token appended to the node embedding set.
Selecting the NULL slot (action index N) signals that the actor refuses
placement for the current VNF. This supports BC pretraining where the greedy
oracle returns z_m=0 (infeasible), and CTDE where refusing is a valid action.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Categorical
from torch_geometric.data import Data

from orion.actors.backbone import GATv2Backbone, GatedGCNBackbone, MLPBackbone
from orion.actors.domain_observation import NODE_FEAT_DIM

VNF_CONTEXT_DIM = 9


class VNFEncoder(nn.Module):
    """Encode VNF context into the backbone's embedding space.

    VNF context [9-dim]:
      0: cpu_demand_norm     (cpu / max_node_cpu in domain)
      1: ram_demand_norm     (ram / max_node_ram in domain)
      2-5: required_tier_onehot  (4-dim)
      6: vcr
      7: bw_demand_norm      (bandwidth / max_link_bw in domain)
      8: position_in_sfc_norm (position / sfc_length)
    """

    def __init__(self, in_dim: int = VNF_CONTEXT_DIM, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, vnf_ctx: Tensor) -> Tensor:
        """Encode VNF context vector.

        Args:
            vnf_ctx: [VNF_CONTEXT_DIM] or [B, VNF_CONTEXT_DIM].

        Returns:
            VNF embedding [out_dim] or [B, out_dim].
        """
        return self.net(vnf_ctx)


class DomainPolicy(nn.Module):
    """Autoregressive pointer-network policy for VNF placement.

    For each VNF in the plan fragment:
      1. Forward pass through GNN backbone -> node embeddings [N, D]
      2. Encode VNF context -> query vector [D]
      3. Dot-product pointer: scores = (node_embeds @ query) / sqrt(D)
      4. Augment with NULL/refuse token (always valid)
      5. Mask invalid real nodes -> softmax -> Categorical over [N+1]
      6. Sample node action (index N = refuse)

    The NULL slot is a learned embedding that competes with real node
    embeddings. Selecting it means the actor refuses placement for this
    VNF (z_m=0 in the BC oracle). During BC pretraining, the NULL label
    is assigned whenever the greedy oracle could not place a VNF.
    """

    NULL_ACTION = -1  # Sentinel returned when the NULL slot is selected

    def __init__(
        self,
        backbone_type: str = "gatv2",
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.backbone_type = backbone_type

        if backbone_type == "gatv2":
            self.backbone = GATv2Backbone(
                in_dim=NODE_FEAT_DIM, hidden_dim=hidden_dim,
                num_heads=num_heads, dropout=dropout,
            )
        elif backbone_type == "gatedgcn":
            self.backbone = GatedGCNBackbone(
                in_dim=NODE_FEAT_DIM, hidden_dim=hidden_dim,
            )
        elif backbone_type == "mlp":
            self.backbone = MLPBackbone(
                in_dim=NODE_FEAT_DIM, hidden_dim=hidden_dim,
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone_type}")

        self.vnf_encoder = VNFEncoder(
            in_dim=VNF_CONTEXT_DIM, out_dim=hidden_dim,
        )

        # Learned NULL/refuse token: starts at zero (neutral)
        self.null_token = nn.Parameter(torch.zeros(hidden_dim))
        # Bias on the NULL logit, initialized negative so untrained actors
        # default to attempting placement; BC training adjusts upward when
        # the greedy oracle shows refusal is correct (z_m=0)
        self.null_bias = nn.Parameter(torch.tensor(-2.0))

        self.scale = math.sqrt(hidden_dim)

    def _encode_and_score(
        self,
        graph_data: Data,
        vnf_context: Tensor,
        action_mask: Tensor,
    ) -> Tensor:
        """Encode graph + VNF and return logits over [N+1] (nodes + NULL).

        The NULL slot (index N) is always unmasked. Real-node slots are
        masked according to action_mask.

        Args:
            graph_data: PyG Data for the domain subgraph.
            vnf_context: VNF context tensor [VNF_CONTEXT_DIM].
            action_mask: Boolean mask [N] for real nodes.

        Returns:
            Logits tensor [N+1].
        """
        node_embeds = self.backbone(graph_data)  # [N, D]
        vnf_embed = self.vnf_encoder(vnf_context)  # [D]

        # Augment node embeddings with the NULL token
        null_embed = self.null_token.unsqueeze(0)  # [1, D]
        h_aug = torch.cat([node_embeds, null_embed], dim=0)  # [N+1, D]

        # Pointer scores: scaled dot-product
        scores = (h_aug @ vnf_embed) / self.scale  # [N+1]

        # Apply learned bias to NULL slot
        scores[-1] = scores[-1] + self.null_bias

        # Mask: real nodes use action_mask, NULL slot is always valid
        aug_mask = torch.cat([action_mask, torch.tensor([True])])
        scores = scores.masked_fill(~aug_mask, float("-inf"))

        return scores

    def forward(
        self,
        graph_data: Data,
        vnf_context: Tensor,
        action_mask: Tensor,
        deterministic: bool = False,
    ) -> tuple[int, Tensor, float]:
        """Score nodes and sample a placement for one VNF.

        The action space is [0..N-1] for real node placements and N for
        the NULL/refuse action. When NULL is selected, this method returns
        NULL_ACTION (-1) as the action index.

        Args:
            graph_data: PyG Data for the domain subgraph (current state).
            vnf_context: VNF context tensor [VNF_CONTEXT_DIM].
            action_mask: Boolean mask [N] where True = valid.
            deterministic: If True, take argmax instead of sampling.

        Returns:
            Tuple of (action_idx, log_prob, entropy).
            action_idx is NULL_ACTION (-1) if the NULL slot was chosen.
        """
        n_nodes = action_mask.size(0)
        scores = self._encode_and_score(graph_data, vnf_context, action_mask)

        dist = Categorical(logits=scores)

        if deterministic:
            action = scores.argmax()
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy().item()

        action_val = action.item()
        if action_val == n_nodes:
            action_val = self.NULL_ACTION

        return action_val, log_prob, entropy

    def forward_bc(
        self,
        graph_data: Data,
        vnf_contexts: Tensor,
    ) -> Tensor:
        """BC pretraining forward: return logits for all VNFs in a fragment.

        Processes all VNFs independently against the same graph state
        (no autoregressive conditioning). This matches the BC training
        setup where the greedy oracle provides per-VNF labels given
        the local state at fragment start.

        Args:
            graph_data: PyG Data for the domain subgraph.
            vnf_contexts: VNF context tensor [K, VNF_CONTEXT_DIM].

        Returns:
            Logits tensor [K, N+1] over (nodes + NULL) for each VNF.
        """
        node_embeds = self.backbone(graph_data)  # [N, D]
        null_embed = self.null_token.unsqueeze(0)  # [1, D]
        h_aug = torch.cat([node_embeds, null_embed], dim=0)  # [N+1, D]

        q = self.vnf_encoder(vnf_contexts)  # [K, D]

        # Batched pointer scores: [K, N+1]
        logits = (q @ h_aug.T) / self.scale

        # Apply learned bias to NULL slot (last column)
        logits[:, -1] = logits[:, -1] + self.null_bias

        return logits
