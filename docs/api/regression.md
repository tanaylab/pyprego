# Regression

Motif discovery via iterative PWM regression. This module contains the core `regress_pwm` function and its variants for multiple motifs, cluster-specific regression, and cross-validation.

::: pyprego.regression
    options:
      members:
        - regress_pwm
        - regress_pwm_core
        - regress_multiple_motifs
        - regress_pwm_clusters
        - regress_pwm_cv
        - MultiRegressionResult
        - ClusterRegressionResult
        - CVRegressionResult
