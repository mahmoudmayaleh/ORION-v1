"""Slice request generation and dataset building."""

from __future__ import annotations

from orion.slicing.request_generator import PoissonArrivalGenerator
from orion.slicing.slice_request import slice_request_from_dict, slice_request_to_dict

__all__ = [
    "PoissonArrivalGenerator",
    "slice_request_to_dict",
    "slice_request_from_dict",
]
