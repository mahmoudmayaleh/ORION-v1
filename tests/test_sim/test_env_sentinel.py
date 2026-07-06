"""The unset-observation-space sentinel must raise on every meaningful access.

The point of the sentinel is to make wrong-shape training crash loudly.
These tests pin that contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from gymnasium import spaces

from orion.sim.env import OrionSliceEnv, _UnsetObservationSpace


class _StubRunner:
    """Minimal runner stand-in — env construction doesn't drive a real episode."""

    def reset(self) -> None:
        pass


@pytest.fixture
def env() -> OrionSliceEnv:
    return OrionSliceEnv(runner=_StubRunner())  # type: ignore[arg-type]


class TestSentinelRaises:
    def test_default_observation_space_is_unset_sentinel(self, env: OrionSliceEnv) -> None:
        assert isinstance(env.observation_space, _UnsetObservationSpace)

    def test_sample_raises(self, env: OrionSliceEnv) -> None:
        with pytest.raises(RuntimeError, match="must probe the runner"):
            env.observation_space.sample()

    def test_contains_raises(self, env: OrionSliceEnv) -> None:
        with pytest.raises(RuntimeError, match="must probe the runner"):
            env.observation_space.contains(np.zeros(4))

    def test_repr_does_not_raise(self, env: OrionSliceEnv) -> None:
        # repr() should be safe for logging — only meaningful operations raise.
        s = repr(env.observation_space)
        assert "Unset" in s

    def test_initial_observation_is_length_zero(self, env: OrionSliceEnv) -> None:
        # The returned observation array is a marker — any caller that
        # ignores observation_space and tries to use [i] will crash.
        assert env._last_obs.shape == (0,)


class TestBindObservationSpace:
    def test_bind_replaces_sentinel(self, env: OrionSliceEnv) -> None:
        real_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        env.bind_observation_space(real_space)
        assert env.observation_space is real_space
        # Sample now works.
        sample = env.observation_space.sample()
        assert sample.shape == (8,)

    def test_bind_resizes_last_obs(self, env: OrionSliceEnv) -> None:
        real_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        env.bind_observation_space(real_space)
        assert env._last_obs.shape == (8,)
        assert env._last_obs.dtype == np.float32
