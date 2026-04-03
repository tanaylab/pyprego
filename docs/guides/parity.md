# R Parity Reference

pyprego is a Python port of the [prego](https://github.com/tanaylab/prego) R package. This page documents the mapping between R and Python functions.

## Function Mapping

### Core Regression

| R function | Python function | Notes |
|---|---|---|
| `regress_pwm()` | `pyprego.regress_pwm()` | Full feature parity |
| `regress_multiple_motifs()` | `pyprego.regress_multiple_motifs()` | Also accessible via `regress_pwm(motif_num=N)` |
| `regress_pwm_clusters()` | `pyprego.regress_pwm_clusters()` | Per-cluster regression |
| `regress_pwm_cv()` | `pyprego.regress_pwm_cv()` | Cross-validated regression |

### PWM Scoring

| R function | Python function | Notes |
|---|---|---|
| `compute_pwm()` | `pyprego.compute_pwm()` | logSumExp and max aggregation |
| `compute_local_pwm()` | `pyprego.compute_local_pwm()` | Per-position scores |

### K-mer Operations

| R function | Python function | Notes |
|---|---|---|
| `screen_kmers()` | `pyprego.screen_kmers()` | Vectorized implementation |
| `generate_kmers()` | `pyprego.generate_kmers()` | Including gapped k-mers |
| `kmer_matrix()` | `pyprego.kmer_matrix()` | K-mer count matrix |
| `kmers_to_pssm()` | `pyprego.kmers_to_pssm()` | K-mer to PSSM conversion |

### PSSM Utilities

| R function | Python function | Notes |
|---|---|---|
| `pssm_cor()` | `pyprego.pssm_cor()` | Best-alignment correlation |
| `pssm_diff()` | `pyprego.pssm_diff()` | Best-alignment distance |
| `pssm_match()` | `pyprego.pssm_match()` | Database matching |
| `pssm_trim()` | `pyprego.pssm_trim()` | Trim low-info edges |
| `pssm_rc()` | `pyprego.pssm_rc()` | Reverse complement |
| `bits_per_pos()` | `pyprego.bits_per_pos()` | Information content |
| `consensus_from_pssm()` | `pyprego.consensus_from_pssm()` | Consensus with IUPAC codes |
| `pssm_quantile()` | `pyprego.pssm_quantile()` | Empirical score quantile |
| `pssm_dataset_cor()` | `pyprego.pssm_dataset_cor()` | Dataset-level correlation |
| `pssm_dataset_diff()` | `pyprego.pssm_dataset_diff()` | Dataset-level distance |

### Motif Database

| R function | Python function | Notes |
|---|---|---|
| `create_motif_db()` | `pyprego.create_motif_db()` | Build MotifDB from sequences |
| `extract_pwm()` | `pyprego.extract_pwm()` | Extract PSSM from MotifDB |
| `screen_pwm()` | `pyprego.screen_pwm()` | Score sequences against all motifs in a DB |
| `motif_enrichment()` | `pyprego.motif_enrichment()` | Motif enrichment analysis |
| `all_motif_datasets()` | `pyprego.all_motif_datasets()` | List bundled datasets |
| `get_motif_pssm()` | `pyprego.get_motif_pssm()` | Retrieve PSSM by name from bundled dataset |

### Visualization

| R function | Python function | Notes |
|---|---|---|
| `plot_pssm_logo()` | `pyprego.plot_pssm_logo()` | Uses logomaker or bar-chart fallback |
| (no direct R equivalent) | `pyprego.plot_spat_model()` | Spatial model visualization |
| (no direct R equivalent) | `pyprego.plot_regression_prediction()` | Prediction scatter plot |
| (no direct R equivalent) | `pyprego.plot_regression_qc()` | Combined QC plot |

### Genomic Integration

| R function | Python function | Notes |
|---|---|---|
| `intervals_to_seq()` | `pyprego.intervals_to_seq()` | Requires pymisha |
| `gextract_pwm()` | `pyprego.gextract_pwm()` | Requires pymisha |
| `gextract_local_pwm()` | `pyprego.gextract_local_pwm()` | Requires pymisha |
| `gextract_pwm_quantile()` | `pyprego.gextract_pwm_quantile()` | Requires pymisha |
| `gintervals_center_by_pssm()` | `pyprego.gintervals_center_by_pssm()` | Requires pymisha |

### Export/Import

| R function | Python function | Notes |
|---|---|---|
| `export_regression_model()` | `pyprego.export_regression_model()` | JSON serialization |
| `load_regression_model()` | `pyprego.load_regression_model()` | JSON deserialization |
| `export_multi_regression()` | `pyprego.export_multi_regression()` | Multi-motif export |
| `load_multi_regression()` | `pyprego.load_multi_regression()` | Multi-motif import |

## Data Structure Mapping

| R | Python | Notes |
|---|---|---|
| Named numeric matrix (PSSM) | `pd.DataFrame` with columns `pos`, `A`, `C`, `G`, `T` | Use `pyprego.pssm_dataframe()` to create |
| Named numeric vector (spatial) | `pd.DataFrame` with columns `bin`, `spat_factor` | Use `pyprego.spatial_dataframe()` to create |
| S4 `MotifDB` object | `pyprego.MotifDB` class | Same stacked-matrix internal representation |
| Named list (regression result) | `pyprego.RegressionResult` dataclass | Has a `.predict()` method |
| Named list (multi result) | `pyprego.MultiRegressionResult` dataclass | Has `.predict()` and `.predict_multi()` methods |

## Parameter Name Differences

Most parameters use the same names as the R package, with underscores replacing dots (Python convention):

| R parameter | Python parameter |
|---|---|
| `motif.length` | `motif_length` |
| `score.metric` | `score_metric` |
| `spat.bin.size` | `spat_bin_size` |
| `spat.num.bins` | `spat_num_bins` |
| `improve.epsilon` | `improve_epsilon` |
| `min.nuc.prob` | `min_nuc_prob` |
| `unif.prior` | `unif_prior` |
| `num.folds` | `num_folds` |
| `log.energy` | `log_energy` |
| `optimize.pwm` | `optimize_pwm` |
| `optimize.spat` | `optimize_spat` |

## Numerical Equivalence

pyprego aims for close numerical agreement with the R prego package. Minor floating-point differences may occur due to:

- Different random number generators (NumPy vs R's RNG).
- Different linear algebra backends (LAPACK implementations).
- Slightly different convergence paths in the coordinate-descent optimizer.

In practice, the discovered motifs are functionally equivalent: the same consensus sequences, comparable R-squared values, and the same biological interpretation.
