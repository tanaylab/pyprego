"""
pyprego - PWM Regression Optimizer for motif discovery in DNA sequences.

Python port of the R prego package. Uses NumPy-based computation for
PSSM scoring, k-mer analysis, and iterative motif regression.
"""

__version__ = "0.0.1"

# --- Core types ---
from .types import (
    NUCLEOTIDES,
    RegressionResult,
    pssm_dataframe,
    pssm_to_array,
    spatial_dataframe,
)

# --- Utilities ---
from .utils import (
    calc_sequences_dinuc_dist,
    calc_sequences_dinucs,
    calc_sequences_trinuc_dist,
    rc,
    rc_array,
    sample_quantile_matched_rows,
    validate_sequences,
)

# --- PSSM operations ---
from .pssm import (
    bits_per_pos,
    concat_pssm,
    consensus_from_pssm,
    pssm_add_prior,
    pssm_concat,
    pssm_cor,
    pssm_dataset_cor,
    pssm_dataset_diff,
    pssm_diff,
    pssm_match,
    pssm_quantile,
    pssm_rc,
    pssm_theoretical_max,
    pssm_theoretical_min,
    pssm_trim,
    trim_pssm,
)

# --- K-mer operations ---
from .kmers import (
    generate_kmers,
    kmer_matrix,
    kmers_to_pssm,
    pssm_to_kmer,
    screen_kmers,
)

# --- PWM computation ---
from .compute import (
    compute_local_pwm,
    compute_pwm,
)

# --- Regression ---
from .regression import (
    CVRegressionResult,
    ClusterRegressionResult,
    MultiRegressionResult,
    regress_multiple_motifs,
    regress_pwm,
    regress_pwm_clusters,
    regress_pwm_core,
    regress_pwm_cv,
)

# --- Export/Load ---
from .export import (
    export_multi_regression,
    export_regression_model,
    load_multi_regression,
    load_regression_model,
)

# --- Motif database ---
from .motif_db import (
    MotifDB,
    all_motif_datasets,
    create_motif_db,
    extract_pwm,
    get_motif_pssm,
    motif_db_to_dataframe,
    motif_enrichment,
    screen_pwm,
    set_prior,
)

# --- Visualization (lazy: matplotlib not imported until needed) ---
from .visualization import (
    plot_pssm_logo,
    plot_regression_prediction,
    plot_regression_prediction_binary,
    plot_regression_qc,
    plot_regression_qc_multi,
    plot_spat_model,
)

# --- Genomic integration (requires pymisha; functions raise ImportError if missing) ---
from .genomic import (
    gextract_local_pwm,
    gextract_pwm,
    gextract_pwm_quantile,
    gintervals_center_by_pssm,
    intervals_to_seq,
)


__all__ = [
    # Version
    "__version__",
    # Types
    "NUCLEOTIDES",
    "RegressionResult",
    "pssm_dataframe",
    "pssm_to_array",
    "spatial_dataframe",
    # Utilities
    "rc",
    "rc_array",
    "validate_sequences",
    "calc_sequences_dinucs",
    "calc_sequences_dinuc_dist",
    "calc_sequences_trinuc_dist",
    "sample_quantile_matched_rows",
    # PSSM
    "bits_per_pos",
    "consensus_from_pssm",
    "concat_pssm",
    "pssm_add_prior",
    "pssm_concat",
    "pssm_cor",
    "pssm_diff",
    "pssm_dataset_cor",
    "pssm_dataset_diff",
    "pssm_match",
    "pssm_quantile",
    "pssm_rc",
    "pssm_theoretical_max",
    "pssm_theoretical_min",
    "pssm_trim",
    "trim_pssm",
    # K-mers
    "generate_kmers",
    "kmer_matrix",
    "kmers_to_pssm",
    "pssm_to_kmer",
    "screen_kmers",
    # Compute
    "compute_local_pwm",
    "compute_pwm",
    # Regression
    "regress_pwm",
    "regress_pwm_core",
    "regress_multiple_motifs",
    "regress_pwm_clusters",
    "regress_pwm_cv",
    "MultiRegressionResult",
    "ClusterRegressionResult",
    "CVRegressionResult",
    # Export/Load
    "export_regression_model",
    "load_regression_model",
    "export_multi_regression",
    "load_multi_regression",
    # Motif DB
    "MotifDB",
    "all_motif_datasets",
    "create_motif_db",
    "extract_pwm",
    "get_motif_pssm",
    "motif_db_to_dataframe",
    "motif_enrichment",
    "screen_pwm",
    "set_prior",
    # Visualization
    "plot_pssm_logo",
    "plot_regression_prediction",
    "plot_regression_prediction_binary",
    "plot_regression_qc",
    "plot_regression_qc_multi",
    "plot_spat_model",
    # Genomic
    "gextract_local_pwm",
    "gextract_pwm",
    "gextract_pwm_quantile",
    "gintervals_center_by_pssm",
    "intervals_to_seq",
]
