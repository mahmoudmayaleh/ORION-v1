"""Domain actor networks for Dec-POMDP multi-domain VNF placement.

Each domain actor is a GATv2-based autoregressive policy that maps a plan
fragment (VNFs assigned to its domain by the MDO) to concrete node placements
and intra-domain flow routes.

Architecture sharing without weight sharing: every actor uses the same GATv2
architecture but maintains its own parameters (Zhong et al. JMLR 2024, HARL).
"""
