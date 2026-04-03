# Getting Started

## Installation

### From PyPI

```bash
pip install pyprego
```

### From source

```bash
git clone https://github.com/tanaylab/pyprego.git
cd pyprego
pip install -e .
```

### Optional dependencies

```bash
pip install pymisha    # Genomic interval integration (misha database)
pip install logomaker  # Sequence logos via logomaker (fallback: bar-chart logos)
pip install matplotlib # Required for any plotting
```

## Basic Motif Discovery

The central function in pyprego is `regress_pwm`. Given a set of equal-length DNA sequences and a numeric response vector, it discovers a position weight matrix (PSSM) and spatial model that best explain the response.

```python
import numpy as np
import pyprego

# Suppose you have sequences and a continuous response
# (e.g. ChIP-seq enrichment, expression level, accessibility score)
sequences = [...]   # list of equal-length DNA strings
response = np.array([...])  # one value per sequence

result = pyprego.regress_pwm(sequences, response)
```

The returned `RegressionResult` contains:

| Attribute   | Description |
|-------------|-------------|
| `pssm`      | PSSM as a pandas DataFrame (columns: pos, A, C, G, T) |
| `spat`      | Spatial model DataFrame (columns: bin, spat_factor) |
| `pred`      | Predicted score for each input sequence |
| `consensus` | Consensus motif string |
| `r2`        | R-squared of prediction vs response |

### Predicting on new sequences

```python
new_scores = result.predict(new_sequences)
```

## Binary Classification

When the response is binary (0/1), use the Kolmogorov-Smirnov metric:

```python
binary_response = np.array([0, 0, 1, 1, 0, 1, ...])

result = pyprego.regress_pwm(
    sequences,
    binary_response,
    score_metric="ks",
)

print(result.ks)    # KS statistic
print(result.pred)  # Predicted scores
```

## PWM Scoring with a Known Motif

If you already have a PSSM (e.g. from a database or a previous run), score sequences directly:

```python
scores = pyprego.compute_pwm(
    sequences,
    result.pssm,
    spat=result.spat,
    bidirect=True,
)
```

For per-position scores along each sequence:

```python
local_scores = pyprego.compute_local_pwm(sequences, result.pssm)
# Returns (n_sequences, n_positions) array
```

## K-mer Screening

Screen k-mers for correlation with the response without running full regression:

```python
kmers_df = pyprego.screen_kmers(sequences, response, kmer_len=8)
print(kmers_df.head(10))
```

The result is a DataFrame sorted by descending `max_r2`, showing correlation between each k-mer's occurrence frequency and the response.

!!! tip "Gapped k-mers"
    Use `max_gap` to include gapped k-mers (positions replaced with `N`):

    ```python
    kmers_df = pyprego.screen_kmers(
        sequences, response, kmer_len=8, max_gap=1
    )
    ```

## Multiple Motifs

To discover more than one motif in the same dataset:

```python
multi = pyprego.regress_multiple_motifs(
    sequences, response, motif_num=3
)

for i, model in enumerate(multi.models):
    print(f"Motif {i+1}: {model.consensus}  (R2={model.r2:.3f})")

print(f"Combined R2: {multi.multi_stats['comb_score'].iloc[-1]:.3f}")
```

## Visualization

```python
# Sequence logo
pyprego.plot_pssm_logo(result.pssm)

# Spatial model
pyprego.plot_spat_model(result.spat)

# Regression QC (prediction vs response)
pyprego.plot_regression_qc(result, response)
```

!!! note "Plotting dependencies"
    All plotting functions require `matplotlib`. The `plot_pssm_logo` function
    uses `logomaker` if available, falling back to a bar-chart representation.

## Exporting and Loading Models

```python
# Save to JSON
pyprego.export_regression_model(result, "model.json")

# Load back
loaded = pyprego.load_regression_model("model.json")
scores = loaded.predict(new_sequences)
```

## Next Steps

- [Motif Discovery Tutorial](../tutorials/motif-discovery.md) for a comprehensive walkthrough.
- [K-mer Analysis Tutorial](../tutorials/kmer-analysis.md) for deeper k-mer exploration.
- [API Reference](../api/index.md) for full function signatures and options.
