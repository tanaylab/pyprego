# Genomic Integration

Functions for working with genomic intervals. All functions in this module require the `pymisha` package.

!!! warning "Optional dependency"
    Install pymisha with `pip install pymisha`. Functions will raise
    `ImportError` with a clear message if pymisha is not available.

::: pyprego.genomic
    options:
      members:
        - intervals_to_seq
        - gextract_pwm
        - gextract_pwm_quantile
        - gextract_local_pwm
        - gintervals_center_by_pssm
