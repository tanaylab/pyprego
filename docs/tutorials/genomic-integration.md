# Genomic Integration

This tutorial covers pyprego's integration with genomic data through the [pymisha](https://github.com/tanaylab/pymisha) package. These functions allow you to extract DNA sequences from genomic intervals and run PWM regression directly on genomic regions.

## Requirements

The genomic module requires `pymisha`:

```bash
pip install pymisha
```

All genomic functions will raise a clear `ImportError` at call time if pymisha is not installed. The rest of pyprego works without it.

!!! note "misha database"
    pymisha requires a configured misha genomic database with a reference genome. See the [pymisha documentation](https://tanaylab.github.io/pymisha/) for setup instructions.

## Extracting Sequences from Genomic Intervals

The `intervals_to_seq` function extracts DNA sequences for a set of genomic intervals:

```python
import pandas as pd
import pyprego

# Define genomic intervals
intervals = pd.DataFrame({
    "chrom": ["chr1", "chr1", "chr2"],
    "start": [1000, 5000, 3000],
    "end":   [1200, 5200, 3200],
})

# Extract sequences (each will be 200 bp)
sequences = pyprego.intervals_to_seq(intervals)
print(len(sequences))     # 3
print(len(sequences[0]))  # 200
```

### Normalizing interval size

If your intervals have variable sizes, normalize them to a fixed width around their centers:

```python
# Intervals of different sizes
intervals = pd.DataFrame({
    "chrom": ["chr1", "chr1", "chr2"],
    "start": [1000, 5000, 3000],
    "end":   [1150, 5300, 3250],
})

# Normalize all to 200 bp
sequences = pyprego.intervals_to_seq(intervals, size=200)
print(len(sequences[0]))  # 200
```

## Running PWM Regression on Genomic Data

### gextract_pwm

`gextract_pwm` is a convenience function that combines sequence extraction and PWM scoring in one step:

```python
import numpy as np

# Genomic intervals with a response column
intervals = pd.DataFrame({
    "chrom": ["chr1"] * 100,
    "start": np.arange(100) * 500 + 10000,
    "end":   np.arange(100) * 500 + 10200,
})
response = np.random.normal(size=100)

# Run PWM regression on genomic intervals
result = pyprego.gextract_pwm(
    intervals,
    response,
    size=200,        # normalize intervals to 200 bp
)

print(f"Consensus: {result.consensus}")
print(f"R-squared: {result.r2:.4f}")
```

Under the hood, `gextract_pwm`:

1. Normalizes intervals to the specified `size` (if given).
2. Extracts DNA sequences via `intervals_to_seq`.
3. Runs `regress_pwm` on the sequences and response.

All keyword arguments are forwarded to `regress_pwm`, so you can use the full set of regression options:

```python
result = pyprego.gextract_pwm(
    intervals,
    response,
    size=300,
    motif_length=12,
    multi_kmers=True,
    score_metric="ks",
    bidirect=True,
)
```

### gextract_pwm_quantile

This function computes PWM score quantiles over genomic intervals, useful for determining score thresholds:

```python
pssm = result.pssm
spat = result.spat

q99 = pyprego.gextract_pwm_quantile(
    intervals,
    pssm,
    spat=spat,
    quantile=0.99,
    size=200,
)
print(f"99th percentile PWM score: {q99:.3f}")
```

### gextract_local_pwm

Compute per-position PWM scores across genomic intervals:

```python
local_scores = pyprego.gextract_local_pwm(
    intervals,
    pssm,
    size=200,
)
print(local_scores.shape)  # (n_intervals, n_positions)
```

## Centering Intervals by Motif Position

`gintervals_center_by_pssm` recenters genomic intervals on the position where a motif scores highest:

```python
centered = pyprego.gintervals_center_by_pssm(
    intervals,
    pssm,
    size=200,
)
print(centered.head())
```

This is useful for aligning peaks or regions around a motif's best binding site, for example before generating aggregate signal profiles.

## Typical Workflow

A common genomic motif analysis workflow:

```python
import pandas as pd
import numpy as np
import pyprego

# 1. Load intervals (e.g. ATAC-seq peaks) with an associated response
peaks = pd.read_csv("peaks.bed", sep="\t",
                     names=["chrom", "start", "end", "score"])
response = peaks["score"].values

intervals = peaks[["chrom", "start", "end"]]

# 2. Extract sequences
sequences = pyprego.intervals_to_seq(intervals, size=300)

# 3. Discover motifs
result = pyprego.regress_pwm(
    sequences, response,
    multi_kmers=True,
    verbose=True,
)

# 4. Match against known motifs
result_matched = pyprego.regress_pwm(
    sequences, response,
    multi_kmers=True,
    match_with_db=True,
)
print(f"Best match: {result_matched.db_match_motif} "
      f"(r={result_matched.db_match_cor:.3f})")

# 5. Visualize
pyprego.plot_pssm_logo(result.pssm)
pyprego.plot_spat_model(result.spat)

# 6. Center intervals on the motif
centered = pyprego.gintervals_center_by_pssm(
    intervals, result.pssm, size=300,
)
```

!!! tip "Performance"
    Sequence extraction can be slow for large interval sets. If you plan to
    run multiple regressions on the same intervals, extract sequences once
    with `intervals_to_seq` and reuse the list.
