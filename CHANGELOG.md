# Changelog

All notable changes to pyprego will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.0.4] - 2026-09-02

### Added

- `calc_freq_local_pwm`: score every motif in a database against a per-position
  base frequency matrix at every start position, returning a motifs x positions
  array. Where `compute_local_pwm` scores one concrete sequence, this scores an
  ensemble summarised by its per-position nucleotide distribution. Two modes:
  `combine="multiply"` (log of the expected likelihood, giving every motif the
  same score on a flat ensemble, so rows are comparable) and `combine="sum"`
  (expected log-likelihood, exact for any joint distribution over positions).
  Both reduce to `compute_local_pwm` when the frequency matrix is one-hot. Port
  of the R prego function of the same name, verified against it to 7.1e-14 on
  both bundled databases, in both modes and both strand settings.
- `n_workers` parameter for `regress_pwm(multi_kmers=True)`, parallelising
  candidate-k-mer evaluation over a thread pool. Defaults to 1, so existing
  calls are unchanged.

### Changed

- `pymisha` and `logomaker` are now core dependencies rather than optional
  extras.

### Removed

- The `genomic`, `viz` and `all` extras. Their contents are now installed by
  default, so `pip install pyprego` covers what `pyprego[all]` used to.

### Dependencies

- `threadpoolctl>=3.0` added, used to keep BLAS single-threaded inside worker
  threads. Falls back to the `OMP_NUM_THREADS` environment variable when absent.

## [0.0.2] - 2026-04-03

### Fixed

- macOS build: disable OpenMP on Darwin (clang lacks native support)
- All ruff lint and format issues resolved

### Added

- CI/CD: lint, test, docs, PyPI publish, conda publish workflows
- MkDocs documentation site with 4 tutorial vignettes
- Shipping script with remote guardrails
- Pre-commit hooks (ruff, trailing whitespace)
- Conda recipe, MANIFEST.in, LICENSE
- README badges (PyPI, CI, Docs, License)

## [0.0.1] - 2025-01-01

### Added

- Initial Python port of the R prego package.
- `regress_pwm` for iterative PWM regression with k-mer seeding.
- `regress_multiple_motifs` for discovering multiple motifs sequentially.
- `regress_pwm_clusters` for cluster-specific motif regression.
- `regress_pwm_cv` for cross-validated regression.
- `compute_pwm` and `compute_local_pwm` for PWM scoring.
- `screen_kmers`, `kmer_matrix`, and `generate_kmers` for k-mer analysis.
- PSSM utilities: `pssm_cor`, `pssm_diff`, `pssm_match`, `pssm_trim`, `pssm_rc`, `bits_per_pos`, `consensus_from_pssm`.
- Motif database support via `MotifDB`, `create_motif_db`, `screen_pwm`, and bundled JASPAR/HOMER datasets.
- Visualization: `plot_pssm_logo`, `plot_spat_model`, `plot_regression_prediction`, `plot_regression_qc`.
- Genomic integration (requires pymisha): `intervals_to_seq`, `gextract_pwm`, `gextract_local_pwm`.
- Model export/import via JSON: `export_regression_model`, `load_regression_model`.
- Optional C extension for vectorized energy computation and k-mer counting.
