"""PPORolloutBuffer tests."""

from __future__ import annotations

import pytest
import torch

from orion.training.buffer import PPORolloutBuffer


def _append(buffer: PPORolloutBuffer, reward: float, done: bool) -> None:
    buffer.append_mdo(
        mdo_obs=torch.zeros(4),
        action=[0, 1],
        log_prob=torch.zeros(2),
        entropy=0.0,
        aux_value=0.0,
        global_state=torch.zeros(8),
        critic_value=0.0,
        reward=reward,
        done=done,
    )


class TestAppendAndStack:
    def test_len_grows_with_appends(self) -> None:
        b = PPORolloutBuffer()
        _append(b, 1.0, False)
        _append(b, 2.0, True)
        assert len(b) == 2

    def test_reward_tensor_shape(self) -> None:
        b = PPORolloutBuffer()
        _append(b, 1.5, False)
        _append(b, 2.5, True)
        r = b.reward_tensor()
        assert r.shape == (2,)
        assert r.tolist() == [1.5, 2.5]

    def test_value_tensor_has_bootstrap_appended(self) -> None:
        b = PPORolloutBuffer()
        _append(b, 1.0, False)
        _append(b, 1.0, False)
        v = b.value_tensor(bootstrap=3.0)
        assert v.shape == (3,)
        assert v[-1].item() == pytest.approx(3.0)


class TestMinibatches:
    def test_minibatch_indices_cover_all_samples(self) -> None:
        b = PPORolloutBuffer()
        for _ in range(7):
            _append(b, 0.0, False)
        seen: set[int] = set()
        for mb in b.minibatches(minibatch_size=3):
            seen.update(mb)
        assert seen == set(range(7))

    def test_empty_buffer_yields_no_minibatches(self) -> None:
        b = PPORolloutBuffer()
        assert list(b.minibatches(64)) == []


class TestClear:
    def test_clear_resets_everything(self) -> None:
        b = PPORolloutBuffer()
        _append(b, 1.0, False)
        b.set_gae(torch.zeros(1), torch.zeros(1))
        b.clear()
        assert len(b) == 0
        assert b.advantages is None
        assert b.returns is None
