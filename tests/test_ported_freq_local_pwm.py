"""Ported from R prego tests/testthat/test-freq-local-pwm.R

Tests for calc_freq_local_pwm: expected local PWM scores over a per-position
base frequency matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pyprego
from pyprego.compute import freq_local_pwm_block_size
from pyprego.motif_db import calc_freq_local_pwm, create_motif_db, motif_db_to_dataframe

NUCS = ["A", "C", "G", "T"]


# ---------------------------------------------------------------------------
# Fixtures: two motifs of different lengths, so padding and the per-motif NaN
# tail are exercised everywhere.
# ---------------------------------------------------------------------------


def _freq_test_db() -> pyprego.MotifDB:
    m4 = pd.DataFrame(
        {
            "motif": "M4",
            "pos": [1, 2, 3, 4],
            "A": [0.7, 0.1, 0.1, 0.25],
            "C": [0.1, 0.7, 0.1, 0.25],
            "G": [0.1, 0.1, 0.7, 0.25],
            "T": [0.1, 0.1, 0.1, 0.25],
        }
    )
    m6 = pd.DataFrame(
        {
            "motif": "M6",
            "pos": [1, 2, 3, 4, 5, 6],
            "A": [0.9, 0.05, 0.4, 0.25, 0.1, 0.6],
            "C": [0.05, 0.9, 0.2, 0.25, 0.1, 0.2],
            "G": [0.03, 0.03, 0.2, 0.25, 0.7, 0.1],
            "T": [0.02, 0.02, 0.2, 0.25, 0.1, 0.1],
        }
    )
    return create_motif_db(pd.concat([m4, m6], ignore_index=True))


def _random_freqs(m: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q = rng.uniform(0.05, 1.0, size=(m, 4))
    return q / q.sum(axis=1, keepdims=True)


def _onehot_freqs(seq: str) -> np.ndarray:
    return np.eye(4)[[NUCS.index(ch) for ch in seq]]


def _brute_freq_local_pwm(q: np.ndarray, mdb: pyprego.MotifDB, combine: str, bidirect: bool) -> np.ndarray:
    """Independent reference: the definition, written as nested loops."""
    lengths = list(mdb.motif_lengths.values())
    n, m = len(lengths), q.shape[0]
    out = np.full((n, m), np.nan)

    def one_strand(mat: np.ndarray, i: int, j: int, length: int) -> float:
        total = 0.0
        for offset in range(length):
            p = mat[4 * offset : 4 * offset + 4, i]
            total += q[j + offset] @ p if combine == "sum" else np.log(q[j + offset] @ np.exp(p))
        return total

    for i, length in enumerate(lengths):
        for j in range(m - length + 1):
            fwd = one_strand(mdb.mat, i, j, length)
            out[i, j] = np.logaddexp(fwd, one_strand(mdb.rc_mat, i, j, length)) if bidirect else fwd
    return out


# ---------------------------------------------------------------------------
# Anchors ported from the R test file
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bidirect", [True, False])
def test_onehot_reproduces_compute_local_pwm(bidirect: bool) -> None:
    """A certain ensemble is just a sequence, and scores like one."""
    mdb = _freq_test_db()
    seq = "ACGTACGTTGCAAGGTCCAT"
    q = _onehot_freqs(seq)
    tidy = motif_db_to_dataframe(mdb)

    result = calc_freq_local_pwm(q, mdb, combine="sum", bidirect=bidirect)
    for i, (motif, length) in enumerate(mdb.motif_lengths.items()):
        pssm = tidy[tidy["motif"] == motif][NUCS].reset_index(drop=True)
        expected = pyprego.compute_local_pwm([seq], pssm, bidirect=bidirect, prior=mdb.prior)[0]
        valid = slice(0, len(seq) - length + 1)
        # R prego's compute_local_pwm computes in float, hence 1e-5.
        np.testing.assert_allclose(result[i, valid], expected[valid], rtol=0, atol=1e-5)


def test_both_combine_methods_agree_on_onehot() -> None:
    mdb = _freq_test_db()
    q = _onehot_freqs("ACGTACGTTGCAAGGTCCAT")
    a = calc_freq_local_pwm(q, mdb, combine="sum", bidirect=True)
    b = calc_freq_local_pwm(q, mdb, combine="multiply", bidirect=True)
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-5, equal_nan=True)


@pytest.mark.parametrize("combine", ["multiply", "sum"])
@pytest.mark.parametrize("bidirect", [True, False])
def test_matches_brute_force(combine: str, bidirect: bool) -> None:
    mdb = _freq_test_db()
    q = _random_freqs(30)
    result = calc_freq_local_pwm(q, mdb, combine=combine, bidirect=bidirect)
    expected = _brute_freq_local_pwm(q, mdb, combine, bidirect)
    np.testing.assert_array_equal(np.isnan(result), np.isnan(expected))
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-12, equal_nan=True)


def test_flat_frequencies_give_every_motif_the_same_floor() -> None:
    """Under "multiply" a flat ensemble scores L * log(0.25), whatever the motif."""
    mdb = _freq_test_db()
    m = 20
    q = np.full((m, 4), 0.25)

    result = calc_freq_local_pwm(q, mdb, combine="multiply", bidirect=False)
    for i, length in enumerate(mdb.motif_lengths.values()):
        valid = slice(0, m - length + 1)
        np.testing.assert_allclose(result[i, valid], length * np.log(0.25), rtol=0, atol=1e-12)

    # ...whereas the expected-log-likelihood floor is motif-dependent.
    res_sum = calc_freq_local_pwm(q, mdb, combine="sum", bidirect=False)
    lengths = list(mdb.motif_lengths.values())
    assert not np.isclose(res_sum[1, 0] / lengths[1], res_sum[0, 0] / lengths[0])


def test_nan_tail_follows_each_motif_length() -> None:
    mdb = _freq_test_db()
    m = 12
    result = calc_freq_local_pwm(_random_freqs(m), mdb)

    assert result.shape == (2, m)
    assert list(mdb.motif_lengths) == ["M4", "M6"]
    for i, length in enumerate(mdb.motif_lengths.values()):
        assert not np.isnan(result[i, : m - length + 1]).any()
        assert np.isnan(result[i, m - length + 1 :]).all()


def test_input_forms_give_identical_results() -> None:
    mdb = _freq_test_db()
    q1, q2 = _random_freqs(15, seed=1), _random_freqs(15, seed=2)
    expected = [calc_freq_local_pwm(q1, mdb), calc_freq_local_pwm(q2, mdb)]

    def check(got: list[np.ndarray]) -> None:
        assert len(got) == 2
        for a, b in zip(got, expected, strict=True):
            np.testing.assert_array_equal(a, b)

    check(calc_freq_local_pwm([q1, q2], mdb))
    # transposed orientation, 4 x m
    check(calc_freq_local_pwm([q1.T, q2.T], mdb))
    # 3-D array, n x m x 4
    check(calc_freq_local_pwm(np.stack([q1, q2]), mdb))
    # 3-D array, n x 4 x m
    check(calc_freq_local_pwm(np.stack([q1.T, q2.T]), mdb))
    # a single matrix comes back unwrapped
    single = calc_freq_local_pwm(q1, mdb)
    assert isinstance(single, np.ndarray)
    np.testing.assert_array_equal(single, expected[0])


def test_accepts_a_tidy_motif_dataframe() -> None:
    mdb = _freq_test_db()
    q = _random_freqs(15)
    np.testing.assert_allclose(
        calc_freq_local_pwm(q, motif_db_to_dataframe(mdb)),
        calc_freq_local_pwm(q, mdb),
        rtol=0,
        atol=1e-10,
        equal_nan=True,
    )


def test_rejects_frequencies_that_are_not_distributions() -> None:
    mdb = _freq_test_db()
    q = _random_freqs(15)
    q[3] = q[3] * 2
    with pytest.raises(ValueError, match="sum to 1"):
        calc_freq_local_pwm(q, mdb)

    with pytest.raises(ValueError, match="4 rows or 4 columns"):
        calc_freq_local_pwm(np.full((5, 5), 0.2), mdb)

    with pytest.raises(ValueError, match="2-D"):
        calc_freq_local_pwm(np.full(4, 0.25), mdb)

    with pytest.raises(ValueError, match="empty"):
        calc_freq_local_pwm([], mdb)


def test_rejects_a_matrix_shorter_than_the_longest_motif() -> None:
    mdb = _freq_test_db()
    with pytest.raises(ValueError, match="shorter than the longest motif"):
        calc_freq_local_pwm(_random_freqs(3), mdb)


def test_rejects_bad_combine_and_motifs() -> None:
    mdb = _freq_test_db()
    q = _random_freqs(15)
    with pytest.raises(ValueError, match="combine must be"):
        calc_freq_local_pwm(q, mdb, combine="product")
    with pytest.raises(ValueError, match="MotifDB object or a tidy motif DataFrame"):
        calc_freq_local_pwm(q, "not a database")


# ---------------------------------------------------------------------------
# Blocking: batch-size invariance and single-matrix parallelism
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("combine", ["multiply", "sum"])
@pytest.mark.parametrize("bidirect", [True, False])
def test_results_are_bit_identical_across_batch_sizes(combine: str, bidirect: bool) -> None:
    """A matrix must score identically however many are passed alongside it.

    The block width is the inner dimension of the per-block matmul, so deriving
    it from the batch size would let BLAS pick a different kernel and shift the
    result by an ULP or two depending on the call.
    """
    db = pyprego.all_motif_datasets()
    names = db["motif"].unique()[:80]
    mdb = create_motif_db(db[db["motif"].isin(names)])
    matrices = [_random_freqs(90, seed=s) for s in range(6)]

    kwargs = {"combine": combine, "bidirect": bidirect, "n_threads": 4}
    reference = calc_freq_local_pwm(matrices[0], mdb, **kwargs)
    for n_batch in (2, 3, 6):
        batched = calc_freq_local_pwm(matrices[:n_batch], mdb, **kwargs)
        np.testing.assert_array_equal(
            reference.view(np.uint64),
            batched[0].view(np.uint64),
            err_msg=f"batch of {n_batch} shifted the first matrix's scores",
        )


def test_block_size_depends_only_on_database_and_threads() -> None:
    # A single matrix must still split into at least n_threads tasks.
    for n_motifs in (61, 500, 3867):
        for n_threads in (1, 8, 32, 64):
            block = freq_local_pwm_block_size(n_motifs, n_threads)
            assert 1 <= block <= 64
            n_blocks = -(-n_motifs // block)
            assert n_blocks >= min(n_threads, -(-n_motifs // 16)), (n_motifs, n_threads, block)


def test_threaded_and_serial_agree() -> None:
    mdb = _freq_test_db()
    matrices = [_random_freqs(40, seed=s) for s in range(3)]
    serial = calc_freq_local_pwm(matrices, mdb, n_threads=1)
    threaded = calc_freq_local_pwm(matrices, mdb, n_threads=8)
    for a, b in zip(serial, threaded, strict=True):
        np.testing.assert_array_equal(a.view(np.uint64), b.view(np.uint64))
