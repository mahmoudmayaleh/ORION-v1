"""Event-driven Poisson arrival process with exponential slice lifetimes.

  - Static offline: batch of S requests at t=0 (for MILP oracle)
  - Dynamic online: Poisson arrivals (rate lambda), Exp lifetimes (rate mu)

Adapted from Virne's VirtualNetworkRequestSimulator event generation pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from orion.substrate.graph_model import SubstrateNetwork
from orion.sim.slice_generator import generate_slice_request
from orion.types import SliceRequest


class EventType(IntEnum):
    ARRIVAL = 1
    DEPARTURE = 0


@dataclass
class SimEvent:
    """A discrete simulation event."""
    time: float
    event_type: EventType
    request_id: str
    slice_request: SliceRequest | None = None  # Only set for ARRIVAL events


class ArrivalProcess:
    """Generates a stream of slice arrival and departure events.

    Events are pre-generated and sorted chronologically, following
    Virne's VirtualNetworkRequestSimulator pattern.

    Attributes:
        events: Sorted list of SimEvents for the episode.
        arrival_rate: Poisson rate lambda (arrivals per time unit).
        service_rate: Exponential rate mu (1/mean_lifetime).
    """

    def __init__(
        self,
        substrate: SubstrateNetwork,
        num_arrivals: int,
        arrival_rate: float,
        service_rate: float,
        rng: np.random.Generator,
    ) -> None:
        self.substrate = substrate
        self.num_arrivals = num_arrivals
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        self.rng = rng
        self.events: list[SimEvent] = []
        self._event_idx: int = 0

    def generate(self) -> None:
        """Generate all arrival and departure events for one episode."""
        events: list[SimEvent] = []

        # Inter-arrival times ~ Exp(lambda)
        inter_arrivals = self.rng.exponential(
            1.0 / self.arrival_rate, size=self.num_arrivals
        )
        arrival_times = np.cumsum(inter_arrivals)

        # Lifetimes ~ Exp(mu)
        lifetimes = self.rng.exponential(
            1.0 / self.service_rate, size=self.num_arrivals
        )

        for i in range(self.num_arrivals):
            t_arrive = float(arrival_times[i])
            t_depart = t_arrive + float(lifetimes[i])
            req_id = f"req_{i:04d}"

            request = generate_slice_request(
                request_id=req_id,
                substrate=self.substrate,
                rng=self.rng,
                arrival_time=t_arrive,
                lifetime=float(lifetimes[i]),
            )

            events.append(SimEvent(
                time=t_arrive,
                event_type=EventType.ARRIVAL,
                request_id=req_id,
                slice_request=request,
            ))
            events.append(SimEvent(
                time=t_depart,
                event_type=EventType.DEPARTURE,
                request_id=req_id,
                slice_request=None,
            ))

        # Sort chronologically; departures before arrivals at same time
        events.sort(key=lambda e: (e.time, e.event_type.value))
        self.events = events
        self._event_idx = 0

    def reset(self) -> None:
        """Re-generate events for a new episode."""
        self.generate()

    def has_next(self) -> bool:
        return self._event_idx < len(self.events)

    def peek(self) -> SimEvent:
        return self.events[self._event_idx]

    def next_event(self) -> SimEvent:
        event = self.events[self._event_idx]
        self._event_idx += 1
        return event

    @property
    def num_events(self) -> int:
        return len(self.events)

    @property
    def num_remaining(self) -> int:
        return len(self.events) - self._event_idx
