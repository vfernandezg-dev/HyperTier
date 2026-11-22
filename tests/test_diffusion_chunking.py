# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for hypertier.diffusion.chunking: plain PyTorch on the CPU, no GPU needed."""

import pytest

from hypertier.diffusion import chunk_tokens, one_sample_at_a_time, runs_one_sample_at_a_time

torch = pytest.importorskip("torch")


class Mixer(torch.nn.Module):
    """A block with every kind of argument: batched, shared by the batch, nested, not a tensor."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.fc = torch.nn.Linear(6, 6)
        self.seen = []

    def forward(self, x, table, pair, *, mask=None, scale=1.0):
        self.seen.append((tuple(x.shape), tuple(table.shape), tuple(pair[1].shape)))
        hidden = self.fc(x + table) * scale + pair[0]
        if mask is not None:
            hidden = hidden * mask
        return hidden, None, [pair[1].sum(dim=1, keepdim=True), scale]


def mixer_arguments(batch):
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(batch, 5, 6, generator=gen)
    table = torch.randn(1, 5, 6, generator=gen)  # one for the whole batch
    pair = (torch.randn(batch, 1, 6, generator=gen), torch.randn(batch, 4, generator=gen))
    mask = torch.rand(batch, 5, 1, generator=gen)
    return x, table, pair, mask


def test_a_batch_runs_as_one_call_per_sample():
    plain, split = Mixer(), Mixer()
    x, table, pair, mask = mixer_arguments(3)
    one_sample_at_a_time(split)
    with torch.no_grad():
        hidden, nothing, (sums, scale) = split(x, table, pair, mask=mask, scale=0.5)
        alone = [
            plain(x[i : i + 1], table, (pair[0][i : i + 1], pair[1][i : i + 1]),
                  mask=mask[i : i + 1], scale=0.5)
            for i in range(3)
        ]  # fmt: skip
    assert split.seen == [((1, 5, 6), (1, 5, 6), (1, 4))] * 3, "the table is passed whole"
    assert torch.equal(hidden, torch.cat([out[0] for out in alone]))
    assert torch.equal(sums, torch.cat([out[2][0] for out in alone]))
    assert nothing is None and scale == 0.5
    assert hidden.shape == (3, 5, 6) and sums.shape == (3, 1)


def test_a_batch_of_one_is_passed_through_untouched():
    split = Mixer()
    x, table, pair, mask = mixer_arguments(1)
    one_sample_at_a_time(split)
    with torch.no_grad():
        hidden, _, (sums, _) = split(x, table, pair, mask=mask)
        expected = Mixer()(x, table, pair, mask=mask)
    assert split.seen == [((1, 5, 6), (1, 5, 6), (1, 4))]
    assert torch.equal(hidden, expected[0]) and torch.equal(sums, expected[2][0])


def test_outputs_have_to_keep_the_batch_dimension():
    class Total(torch.nn.Module):
        def forward(self, x):
            return x.sum()

    module = Total()
    one_sample_at_a_time(module)
    assert float(module(torch.ones(1, 3))) == 3.0
    with pytest.raises(ValueError, match="Total returned .* must keep the batch dimension"):
        module(torch.ones(2, 3))


def test_undoing_gives_the_forward_of_the_class_back():
    module = Mixer()
    assert not runs_one_sample_at_a_time(module)
    undo = one_sample_at_a_time(module)
    assert runs_one_sample_at_a_time(module) and "forward" in vars(module)
    undo()
    undo()  # idempotent
    assert not runs_one_sample_at_a_time(module) and "forward" not in vars(module)


def test_replacements_on_one_module_are_undone_in_reverse_order():
    module = torch.nn.Linear(6, 6)
    first = one_sample_at_a_time(module)
    second = chunk_tokens(module, 2)
    assert not runs_one_sample_at_a_time(module), "the outermost replacement is what runs"
    with pytest.raises(RuntimeError, match="in reverse order"):
        first()
    second()
    assert runs_one_sample_at_a_time(module)
    first()
    assert "forward" not in vars(module)


def test_tokens_run_in_chunks_with_the_other_arguments_whole():
    class PerToken(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            torch.manual_seed(0)
            self.net = torch.nn.Sequential(
                torch.nn.Linear(6, 24), torch.nn.GELU(), torch.nn.Linear(24, 3)
            )
            self.seen = []

        def forward(self, x, gain, *, shift=0.0):
            self.seen.append(x.shape[1])
            return self.net(x) * gain + shift

    plain, chunked = PerToken(), PerToken()
    undo = chunk_tokens(chunked, 4)
    x = torch.randn(2, 10, 6, generator=torch.Generator().manual_seed(2))
    gain = torch.tensor(1.5)
    with torch.no_grad():
        got = chunked(x, gain, shift=0.25)
        expected = plain(x, gain, shift=0.25)
        assert chunked.seen == [4, 4, 2]
        assert got.shape == expected.shape == (2, 10, 3)
        # Rounding only: another number of rows may pick another kernel.
        assert torch.allclose(got, expected, rtol=1e-5, atol=1e-6)
        chunked.seen.clear()
        chunked(x[:, :4], gain)
        assert chunked.seen == [4], "what fits in one chunk is one call"
    undo()
    assert "forward" not in vars(chunked)
    with pytest.raises(ValueError, match="0 tokens at a time"):
        chunk_tokens(plain, 0)
