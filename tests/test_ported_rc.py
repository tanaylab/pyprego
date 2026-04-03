"""Ported from R prego tests/testthat/test-rc.R

Tests for the reverse complement function.
"""

from __future__ import annotations

import pytest

import pyprego


class TestRcBasic:
    """rc function works correctly (from R test_that block)."""

    def test_single_sequence(self):
        assert pyprego.rc("ATCG") == "CGAT"

    def test_multiple_sequences(self):
        inputs = ["ATCG", "GGCC", "TATA"]
        expected = ["CGAT", "GGCC", "TATA"]
        results = pyprego.rc_array(inputs)
        assert results == expected

    def test_lowercase_input(self):
        # R rc("atcg") returns "CGAT" (uppercased)
        # Python rc preserves case: rc("atcg") == "cgat"
        result = pyprego.rc("atcg")
        assert result.upper() == "CGAT"

    def test_mixed_case_input(self):
        # R rc("AtCg") returns "CGAT"
        # Python preserves case in complement, so we compare uppercased
        result = pyprego.rc("AtCg")
        assert result.upper() == "CGAT"

    def test_empty_string(self):
        assert pyprego.rc("") == ""

    def test_vector_with_empty_string(self):
        results = pyprego.rc_array(["ATCG", "", "GGCC"])
        assert results == ["CGAT", "", "GGCC"]

    def test_long_sequence(self):
        long_seq = "ATCG" * 1000
        expected = "CGAT" * 1000
        assert pyprego.rc(long_seq) == expected

    def test_palindrome(self):
        # ACGT is its own reverse complement
        assert pyprego.rc("ACGT") == "ACGT"

    def test_all_same_base(self):
        assert pyprego.rc("AAAA") == "TTTT"
        assert pyprego.rc("CCCC") == "GGGG"
        assert pyprego.rc("GGGG") == "CCCC"
        assert pyprego.rc("TTTT") == "AAAA"

    def test_n_base(self):
        assert pyprego.rc("ANCG") == "CGNT"


class TestRcErrors:
    """rc function handles errors correctly (from R test_that block)."""

    def test_list_input_raises(self):
        # R: expect_error(rc(list("ATCG")), "The input should be a character vector")
        # Python: rc expects str, not list
        with pytest.raises(TypeError):
            pyprego.rc(["ATCG"])  # type: ignore[arg-type]

    def test_numeric_input_raises(self):
        with pytest.raises(TypeError):
            pyprego.rc(123)  # type: ignore[arg-type]

    def test_invalid_characters_raises(self):
        # Python rc raises ValueError for non-DNA characters
        with pytest.raises(ValueError):
            pyprego.rc("AT-CG")


class TestRcArray:
    """Test rc_array for vector operations."""

    def test_basic_array(self):
        seqs = ["ATCG", "GGCC"]
        expected = ["CGAT", "GGCC"]
        assert pyprego.rc_array(seqs) == expected

    def test_single_element(self):
        assert pyprego.rc_array(["ACGT"]) == ["ACGT"]

    def test_empty_array(self):
        assert pyprego.rc_array([]) == []
