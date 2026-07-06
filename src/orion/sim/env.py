"""Thin Gymnasium wrapper around `EpisodeRunner`.

The real training interface is `EpisodeRunner` (imperative, multi-agent).
This wrapper exists for tooling that expects the Gym API:
  - SB3 EvalCallback / VecEnv-based evaluation tools
  - Tensorboard monitors
  - Generic dataset collection

It is **not** the path the MAPPO training loop takes. MAPPO calls
`EpisodeRunner.run_episode()` directly so it can read per-agent rollouts.

Action space note: from the Gym API perspective, the "action" is a no-op.
The MDO policy and domain actors live inside the coordinator that the
runner owns. We expose a `Discrete(1)` action space to keep the API
honest. Callers that wish to swap policies do so by constructing the
EpisodeRunner with a different coordinator before wrapping.

Observation space note: the real MDO-observation tensor shape depends on
the substrate (num_domains × per-domain-feat + inter-domain-link-count ×
link-feat + plan-feat + retry-stats). Phase 5's trainer must probe the
runner once at episode start and replace `observation_space` with a real
`Box`. Until that happens we install `_UnsetObservationSpace`, a sentinel
that raises on any attempt to sample, contain, or inspect it. The point
is: if something reads `env.observation_space.shape` thinking it's a real
Box(64,) and starts allocating tensors of that shape, the silent wrong
answer is worse than a loud crash. The sentinel makes the wrong path
crash, while leaving the env usable for non-observation tooling (rewards,
stats, episode lengths).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from orion.sim.episode_runner import EpisodeRunner, EpisodeStats


class _UnsetObservationSpace(gym.spaces.Space):
    """Sentinel observation space. Raises on every meaningful access.

    Replace with a real Box (or Dict) after probing the runner's MDO
    observation tensor shape — see `OrionSliceEnv.bind_observation_space`.
    """

    _MSG = (
        "OrionSliceEnv.observation_space is an unset sentinel. The real MDO "
        "observation shape depends on the substrate; the Phase 5 trainer "
        "must probe the runner and call env.bind_observation_space(box) "
        "before any code that reads .shape / .sample() / .contains() runs. "
        "Reading from the sentinel is a bug — fail loudly rather than train "
        "on the wrong tensor shape."
    )

    def __init__(self) -> None:
        super().__init__(shape=None, dtype=None)

    def sample(self, mask: Any = None, probability: Any = None) -> Any:
        raise RuntimeError(self._MSG)

    def contains(self, x: Any) -> bool:
        raise RuntimeError(self._MSG)

    def __repr__(self) -> str:
        return "<UnsetObservationSpace: must be bound before use>"


class OrionSliceEnv(gym.Env):
    """Per-arrival Gymnasium env over an episode of slice arrivals.

    Each `step()` advances one ARRIVAL event (departures are absorbed
    silently because they generate no reward). The episode terminates when
    the arrival process is drained (Choice F1).

    The observation returned by `reset()` / `step()` is a length-0 float32
    array — a marker that no real observation tensor has been bound. The
    env is fully usable for tooling that only consumes rewards and infos.
    Anything that reads from `observation_space` will raise until
    `bind_observation_space()` is called by the Phase 5 trainer.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        runner: EpisodeRunner,
        mdo_mode: str = "sample",
    ) -> None:
        super().__init__()
        self.runner = runner
        self.mdo_mode = mdo_mode

        # Action is a no-op; the runner owns the policy.
        self.action_space = spaces.Discrete(1)

        # Observation space deferred — sentinel that raises on access.
        self.observation_space = _UnsetObservationSpace()

        self._last_stats = EpisodeStats()
        # Length-0 marker: any caller that ignores .observation_space and
        # tries to use this array will crash on the first indexing attempt.
        self._last_obs = np.empty(0, dtype=np.float32)

    def bind_observation_space(self, space: gym.spaces.Space) -> None:
        """Install the real observation space (call once from Phase 5 trainer).

        Resizes `_last_obs` to match the bound shape so step/reset start
        returning correctly-shaped zero observations until a real obs is
        produced.
        """
        self.observation_space = space
        if isinstance(space, spaces.Box) and space.shape is not None:
            self._last_obs = np.zeros(space.shape, dtype=space.dtype or np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.runner.reset()
        self._last_stats = EpisodeStats()
        # If a real observation space was bound, _last_obs is sized already
        # (set in bind_observation_space). If still on the sentinel, leave
        # the length-0 marker; touching it downstream will crash, which is
        # the point.
        return self._last_obs, {}

    def step(
        self, action: Any  # noqa: ARG002 — Gym requires it; runner owns the policy
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Drain to the next arrival, run one MDO decision, return reward.

        Returns the standard 5-tuple. The reward is the final R_t for the
        arrival just processed (0.0 if the episode ended on a departure or
        was already terminated).
        """
        # Advance through any leading departures.
        while self.runner.arrival_process.has_next():
            evt = self.runner.arrival_process.peek()
            if evt.event_type.value == 0:  # DEPARTURE
                self.runner.arrival_process.next_event()
                self.runner._handle_departure(evt.request_id, self._last_stats)
                continue
            break

        if not self.runner.arrival_process.has_next():
            return self._last_obs, 0.0, True, False, {"stats": self._last_stats}

        event = self.runner.arrival_process.next_event()
        assert event.slice_request is not None

        prev_cum_reward = self._last_stats.cumulative_reward
        from orion.sim.rollout_buffer import MultiAgentRollout

        scratch_rollout = MultiAgentRollout()
        scratch_results: list = []

        self.runner._handle_arrival(
            event.slice_request,
            self.mdo_mode,
            scratch_rollout,
            scratch_results,
            self._last_stats,
        )

        step_reward = self._last_stats.cumulative_reward - prev_cum_reward
        terminated = not self.runner.arrival_process.has_next()
        info = {
            "stats": self._last_stats,
            "rollout": scratch_rollout,
            "mdo_result": scratch_results[0] if scratch_results else None,
        }
        return self._last_obs, float(step_reward), terminated, False, info
