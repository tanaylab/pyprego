# Changelog

All notable changes to pyprego will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
