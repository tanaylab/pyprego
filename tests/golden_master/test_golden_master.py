"""Golden master tests: compare pyprego output against R prego reference data.

Each test loads precomputed R results from JSON files and verifies that the
corresponding pyprego function produces equivalent output (within tolerance).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pyprego


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rows_to_pssm(rows: list[dict]) -> pd.DataFrame:
    """Convert row-oriented PSSM JSON to a pyprego PSSM DataFrame."""
    df = pd.DataFrame(rows)
    cols = []
    if "pos" in df.columns:
        cols.append("pos")
    cols.extend(["A", "C", "G", "T"])
    return df[cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# (a) reverse complement
# ---------------------------------------------------------------------------


class TestRC:
    def test_rc_matches_r(self, golden_rc: dict) -> None:
        for inp, expected in zip(golden_rc["inputs"], golden_rc["outputs"]):
            assert pyprego.rc(inp) == expected, f"rc({inp!r}) mismatch"


# ---------------------------------------------------------------------------
# (b) bits_per_pos
# ---------------------------------------------------------------------------


class TestBitsPerPos:
    def test_bits_per_pos_matches_r(self, golden_bits_per_pos: dict) -> None:
        pssm = _rows_to_pssm(golden_bits_per_pos["pssm"])
        prior = golden_bits_per_pos["prior"]
        expected = np.array(golden_bits_per_pos["bits"])

        result = pyprego.bits_per_pos(pssm, prior=prior)
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-10)


# ---------------------------------------------------------------------------
# (c) consensus_from_pssm
# ---------------------------------------------------------------------------


class TestConsensus:
    def test_consensus_matches_r(self, golden_consensus: dict) -> None:
        pssm = _rows_to_pssm(golden_consensus["pssm"])
        # R uses single_thresh=0.4, double_thresh=0.6 by default
        result = pyprego.consensus_from_pssm(
            pssm,
            single_thresh=golden_consensus["single_thresh"],
            double_thresh=golden_consensus["double_thresh"],
        )
        expected = golden_consensus["consensus"]
        assert result == expected, f"consensus mismatch: {result!r} != {expected!r}"


# ---------------------------------------------------------------------------
# (d) pssm_trim
# ---------------------------------------------------------------------------


class TestPssmTrim:
    def test_pssm_trim_matches_r(self, golden_pssm_trim: dict) -> None:
        pssm = _rows_to_pssm(golden_pssm_trim["input_pssm"])
        bits_thresh = golden_pssm_trim["bits_thresh"]
        expected_df = _rows_to_pssm(golden_pssm_trim["trimmed_pssm"])

        result = pyprego.pssm_trim(pssm, bits_thresh=bits_thresh)

        # Compare dimensions
        assert len(result) == golden_pssm_trim["trimmed_nrow"], (
            f"trimmed length mismatch: {len(result)} != {golden_pssm_trim['trimmed_nrow']}"
        )

        # Compare PSSM values
        for col in ["A", "C", "G", "T"]:
            np.testing.assert_allclose(
                result[col].to_numpy(),
                expected_df[col].to_numpy(),
                rtol=1e-4,
                atol=1e-10,
                err_msg=f"pssm_trim column {col} mismatch",
            )


# ---------------------------------------------------------------------------
# (e) pssm_rc
# ---------------------------------------------------------------------------


class TestPssmRC:
    def test_pssm_rc_matches_r(self, golden_pssm_rc: dict) -> None:
        pssm = _rows_to_pssm(golden_pssm_rc["input_pssm"])
        expected_df = _rows_to_pssm(golden_pssm_rc["rc_pssm"])

        result = pyprego.pssm_rc(pssm)

        for col in ["A", "C", "G", "T"]:
            np.testing.assert_allclose(
                result[col].to_numpy(),
                expected_df[col].to_numpy(),
                rtol=1e-6,
                atol=1e-10,
                err_msg=f"pssm_rc column {col} mismatch",
            )


# ---------------------------------------------------------------------------
# (f) pssm_theoretical_max, pssm_theoretical_min, pssm_quantile
# ---------------------------------------------------------------------------


class TestPssmTheoretical:
    def test_theoretical_max(self, golden_pssm_theoretical: dict) -> None:
        pssm = _rows_to_pssm(golden_pssm_theoretical["pssm"])
        prior = golden_pssm_theoretical["prior"]
        reg = golden_pssm_theoretical["regularization"]
        expected = golden_pssm_theoretical["theoretical_max"]

        result = pyprego.pssm_theoretical_max(pssm, prior=prior, regularization=reg)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_theoretical_min(self, golden_pssm_theoretical: dict) -> None:
        pssm = _rows_to_pssm(golden_pssm_theoretical["pssm"])
        prior = golden_pssm_theoretical["prior"]
        reg = golden_pssm_theoretical["regularization"]
        expected = golden_pssm_theoretical["theoretical_min"]

        result = pyprego.pssm_theoretical_min(pssm, prior=prior, regularization=reg)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_quantile_50(self, golden_pssm_theoretical: dict) -> None:
        pssm = _rows_to_pssm(golden_pssm_theoretical["pssm"])
        prior = golden_pssm_theoretical["prior"]
        reg = golden_pssm_theoretical["regularization"]
        expected = golden_pssm_theoretical["quantile_50"]

        result = pyprego.pssm_quantile(pssm, q=0.5, prior=prior, regularization=reg)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_quantile_85(self, golden_pssm_theoretical: dict) -> None:
        pssm = _rows_to_pssm(golden_pssm_theoretical["pssm"])
        prior = golden_pssm_theoretical["prior"]
        reg = golden_pssm_theoretical["regularization"]
        expected = golden_pssm_theoretical["quantile_85"]

        result = pyprego.pssm_quantile(pssm, q=0.85, prior=prior, regularization=reg)
        np.testing.assert_allclose(result, expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# (g) pssm_cor
# ---------------------------------------------------------------------------


class TestPssmCor:
    def test_spearman(self, golden_pssm_cor: dict) -> None:
        pssm1 = _rows_to_pssm(golden_pssm_cor["pssm1"])
        pssm2 = _rows_to_pssm(golden_pssm_cor["pssm2"])
        expected = golden_pssm_cor["cor_spearman"]

        result = pyprego.pssm_cor(pssm1, pssm2, method="spearman", prior=golden_pssm_cor["prior"])
        np.testing.assert_allclose(result, expected, rtol=1e-3)

    def test_pearson(self, golden_pssm_cor: dict) -> None:
        pssm1 = _rows_to_pssm(golden_pssm_cor["pssm1"])
        pssm2 = _rows_to_pssm(golden_pssm_cor["pssm2"])
        expected = golden_pssm_cor["cor_pearson"]

        result = pyprego.pssm_cor(pssm1, pssm2, method="pearson", prior=golden_pssm_cor["prior"])
        np.testing.assert_allclose(result, expected, rtol=1e-3)


# ---------------------------------------------------------------------------
# (h) pssm_diff (KL divergence)
# ---------------------------------------------------------------------------


class TestPssmDiff:
    def test_kl_divergence(self, golden_pssm_diff: dict) -> None:
        pssm1 = _rows_to_pssm(golden_pssm_diff["pssm1"])
        pssm2 = _rows_to_pssm(golden_pssm_diff["pssm2"])
        expected = golden_pssm_diff["kl_divergence"]

        result = pyprego.pssm_diff(pssm1, pssm2, prior=golden_pssm_diff["prior"])
        np.testing.assert_allclose(result, expected, rtol=1e-3)


# ---------------------------------------------------------------------------
# (i) generate_kmers
# ---------------------------------------------------------------------------


class TestGenerateKmers:
    def test_kmer_set_matches_r(self, golden_generate_kmers: dict) -> None:
        k = golden_generate_kmers["k"]
        expected_set = set(golden_generate_kmers["kmers"])
        result = pyprego.generate_kmers(k)
        result_set = set(result)

        assert result_set == expected_set, (
            f"generate_kmers({k}) set mismatch: "
            f"missing={expected_set - result_set}, extra={result_set - expected_set}"
        )

    def test_kmer_count(self, golden_generate_kmers: dict) -> None:
        k = golden_generate_kmers["k"]
        expected_n = golden_generate_kmers["n_kmers"]
        result = pyprego.generate_kmers(k)
        assert len(result) == expected_n


# ---------------------------------------------------------------------------
# (j) calc_sequences_dinucs
# ---------------------------------------------------------------------------


class TestCalcSequencesDinucs:
    def test_dinucs_match_r(self, golden_dinucs: dict) -> None:
        sequences = golden_dinucs["sequences"]
        expected_df = pd.DataFrame(golden_dinucs["matrix"])
        expected_colnames = golden_dinucs["colnames"]

        result = pyprego.calc_sequences_dinucs(sequences)

        # The R result is a named matrix; Python returns an ndarray
        # Column order should match the standard dinucleotide order
        from pyprego.utils import dinuc_names

        py_colnames = dinuc_names()

        # Compare values column by column using the colnames from R
        for col_name in expected_colnames:
            r_vals = expected_df[col_name].to_numpy()
            py_col_idx = py_colnames.index(col_name)
            py_vals = result[:, py_col_idx]
            np.testing.assert_array_equal(
                py_vals,
                r_vals,
                err_msg=f"dinucs column {col_name} mismatch",
            )


# ---------------------------------------------------------------------------
# (k) compute_pwm
# ---------------------------------------------------------------------------


class TestComputePwm:
    def test_compute_pwm_default(self, golden_compute_pwm: dict, golden_example_data: dict) -> None:
        pssm = _rows_to_pssm(golden_compute_pwm["pssm"])
        sequences = golden_example_data["sequences"]
        expected = np.array(golden_compute_pwm["scores_default"])

        result = pyprego.compute_pwm(
            sequences,
            pssm,
            bidirect=True,
            prior=golden_compute_pwm["prior"],
            func="logSumExp",
        )

        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-6)

    def test_compute_pwm_max(self, golden_compute_pwm: dict, golden_example_data: dict) -> None:
        pssm = _rows_to_pssm(golden_compute_pwm["pssm"])
        sequences = golden_example_data["sequences"]
        expected = np.array(golden_compute_pwm["scores_max"])

        result = pyprego.compute_pwm(
            sequences,
            pssm,
            bidirect=True,
            prior=golden_compute_pwm["prior"],
            func="max",
        )

        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-6)

    def test_compute_pwm_norc(self, golden_compute_pwm: dict, golden_example_data: dict) -> None:
        pssm = _rows_to_pssm(golden_compute_pwm["pssm"])
        sequences = golden_example_data["sequences"]
        expected = np.array(golden_compute_pwm["scores_norc"])

        result = pyprego.compute_pwm(
            sequences,
            pssm,
            bidirect=False,
            prior=golden_compute_pwm["prior"],
            func="logSumExp",
        )

        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-6)


# ---------------------------------------------------------------------------
# (l) kmer_matrix
# ---------------------------------------------------------------------------


class TestKmerMatrix:
    def test_kmer_matrix_matches_r(self, golden_kmer_matrix: dict) -> None:
        sequences = golden_kmer_matrix["sequences"]
        kmer_length = golden_kmer_matrix["kmer_length"]

        # Get the full R kmer matrix column names
        r_colnames = golden_kmer_matrix["colnames"]
        # The subset has specific columns with values
        r_subset_colnames = golden_kmer_matrix["matrix_subset_colnames"]
        r_subset_df = pd.DataFrame(golden_kmer_matrix["matrix_subset"])

        # Run Python
        py_result = pyprego.kmer_matrix(sequences, kmer_length)

        # Verify dimensions
        expected_dim = golden_kmer_matrix["full_dim"]
        assert py_result.shape[0] == expected_dim[0], f"row count mismatch"
        assert py_result.shape[1] == expected_dim[1], f"col count mismatch"

        # Compare values for the subset columns
        for col_name in r_subset_colnames:
            r_vals = r_subset_df[col_name].to_numpy()
            assert col_name in py_result.columns, f"Missing column {col_name}"
            py_vals = py_result[col_name].to_numpy()
            np.testing.assert_array_equal(
                py_vals,
                r_vals,
                err_msg=f"kmer_matrix column {col_name} mismatch",
            )


# ---------------------------------------------------------------------------
# (m) screen_kmers
# ---------------------------------------------------------------------------


class TestScreenKmers:
    def test_screen_kmers_top_kmers(self, golden_screen_kmers: dict, golden_example_data: dict) -> None:
        """Check that the top-correlated kmers from Python overlap heavily with R results.

        screen_kmers uses C++ in R, so exact kmer ordering may differ slightly
        due to floating point differences. We check that the top N kmers from R
        appear in Python results and have similar correlation values.
        """
        sequences = golden_example_data["sequences"]
        resp_arr = pd.DataFrame(golden_example_data["response_mat"]).to_numpy()
        kmer_length = golden_screen_kmers["kmer_length"]
        min_cor = golden_screen_kmers["min_cor"]

        # Run Python screen_kmers
        py_result = pyprego.screen_kmers(
            sequences,
            resp_arr,
            kmer_len=kmer_length,
            min_cor=min_cor,
            seed=golden_screen_kmers["seed"],
        )

        # Load R results
        r_result = pd.DataFrame(golden_screen_kmers["result"])

        # Both should find kmers
        assert len(py_result) > 0, "Python found no kmers"
        assert len(r_result) > 0, "R found no kmers"

        # Compare top-20 kmers by max_r2 -- check overlap
        r_top = set(r_result.head(20)["kmer"].tolist())
        py_top = set(py_result.head(20)["kmer"].tolist())
        overlap = r_top & py_top
        overlap_frac = len(overlap) / len(r_top) if r_top else 0

        assert overlap_frac >= 0.5, (
            f"Top-20 kmer overlap too low: {overlap_frac:.0%} "
            f"(R top: {r_top}, Py top: {py_top})"
        )

    def test_screen_kmers_correlations(self, golden_screen_kmers: dict, golden_example_data: dict) -> None:
        """For shared kmers, check that correlations are close."""
        sequences = golden_example_data["sequences"]
        resp_arr = pd.DataFrame(golden_example_data["response_mat"]).to_numpy()
        kmer_length = golden_screen_kmers["kmer_length"]
        min_cor = golden_screen_kmers["min_cor"]

        py_result = pyprego.screen_kmers(
            sequences,
            resp_arr,
            kmer_len=kmer_length,
            min_cor=min_cor,
            seed=golden_screen_kmers["seed"],
        )

        r_result = pd.DataFrame(golden_screen_kmers["result"])

        # Find shared kmers and compare max_r2
        r_kmer_r2 = dict(zip(r_result["kmer"], r_result["max_r2"]))
        py_kmer_r2 = dict(zip(py_result["kmer"], py_result["max_r2"]))

        shared = set(r_kmer_r2) & set(py_kmer_r2)
        assert len(shared) > 10, f"Too few shared kmers: {len(shared)}"

        r_vals = np.array([r_kmer_r2[k] for k in shared])
        py_vals = np.array([py_kmer_r2[k] for k in shared])

        # Correlations should be very close for shared kmers
        np.testing.assert_allclose(py_vals, r_vals, rtol=0.05, atol=0.01)


# ---------------------------------------------------------------------------
# (r) calc_freq_local_pwm
# ---------------------------------------------------------------------------


def _rows_to_matrix(rows: list[dict]) -> np.ndarray:
    """Convert R's row-oriented data.frame JSON to a 2-D float array.

    R writes NA as JSON null; the column order is the ``V1..Vn`` order the
    data.frame was created with, and ``_row`` carries the row names.
    """
    cols = [k for k in rows[0] if k != "_row"]
    return np.array([[np.nan if r[c] is None else float(r[c]) for c in cols] for r in rows], dtype=np.float64)


def _rows_to_motif_df(rows: list[dict]) -> pd.DataFrame:
    """Convert a tidy motif data.frame JSON to a DataFrame."""
    return pd.DataFrame(rows)[["motif", "pos", "A", "C", "G", "T"]]


class TestCalcFreqLocalPWM:
    """calc_freq_local_pwm against R prego reference output."""

    MODES = [
        ("multiply", True, "multiply_bidirect"),
        ("multiply", False, "multiply_forward"),
        ("sum", True, "sum_bidirect"),
        ("sum", False, "sum_forward"),
    ]

    @pytest.mark.parametrize("db_key,q_key,out_key", [
        ("small_db", "q_random", "small_random"),
        ("small_db", "q_onehot", "small_onehot"),
        ("wide_db", "q_random", "wide_random"),
    ])
    def test_matches_r(self, golden_freq_local_pwm: dict, db_key: str, q_key: str, out_key: str) -> None:
        gm = golden_freq_local_pwm
        mdb = pyprego.create_motif_db(_rows_to_motif_df(gm[db_key]), prior=gm["prior"])
        q = _rows_to_matrix(gm[q_key])

        for combine, bidirect, key in self.MODES:
            expected = _rows_to_matrix(gm[out_key][key])
            result = pyprego.calc_freq_local_pwm(q, mdb, combine=combine, bidirect=bidirect)

            assert result.shape == expected.shape
            # The NaN tail must land on exactly the same entries.
            np.testing.assert_array_equal(np.isnan(result), np.isnan(expected))
            np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10, equal_nan=True)

    def test_flat_matches_r(self, golden_freq_local_pwm: dict) -> None:
        gm = golden_freq_local_pwm
        mdb = pyprego.create_motif_db(_rows_to_motif_df(gm["small_db"]), prior=gm["prior"])
        q = np.full((gm["q_flat_positions"], 4), 0.25)

        for combine, bidirect, key in self.MODES:
            expected = _rows_to_matrix(gm["small_flat"][key])
            result = pyprego.calc_freq_local_pwm(q, mdb, combine=combine, bidirect=bidirect)
            np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10, equal_nan=True)

    def test_onehot_matches_compute_local_pwm(self, golden_freq_local_pwm: dict) -> None:
        """A certain ensemble scores exactly like the sequence it encodes.

        R's compute_local_pwm uses float internally, hence the looser tolerance.
        """
        gm = golden_freq_local_pwm
        mdb = pyprego.create_motif_db(_rows_to_motif_df(gm["small_db"]), prior=gm["prior"])
        q = _rows_to_matrix(gm["q_onehot"])
        lengths = np.array(gm["small_lengths"])

        for bidirect, key in ((True, "bidirect"), (False, "forward")):
            expected = _rows_to_matrix(gm["compute_local_pwm_onehot"][key])
            for combine in ("multiply", "sum"):
                result = pyprego.calc_freq_local_pwm(q, mdb, combine=combine, bidirect=bidirect)
                for i, length in enumerate(lengths):
                    valid = slice(0, q.shape[0] - int(length) + 1)
                    np.testing.assert_allclose(result[i, valid], expected[i, valid], rtol=0, atol=1e-5)
