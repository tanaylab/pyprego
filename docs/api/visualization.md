# Visualization

Sequence logos, spatial model plots, and regression diagnostic figures.

!!! note "Dependencies"
    All functions require `matplotlib`. The `plot_pssm_logo` function uses
    `logomaker` if available, falling back to a bar-chart representation.

::: pyprego.visualization
    options:
      members:
        - plot_pssm_logo
        - plot_spat_model
        - plot_regression_prediction
        - plot_regression_prediction_binary
        - plot_regression_qc
        - plot_regression_qc_multi
