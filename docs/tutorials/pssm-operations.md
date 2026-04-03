# Working with PSSMs

This tutorial covers the PSSM (Position-Specific Scoring Matrix) data structure and the utility functions for manipulating, comparing, and analysing motif weight matrices.

## PSSM Structure

In pyprego, a PSSM is a pandas DataFrame with columns `pos`, `A`, `C`, `G`, `T`:

```python
import numpy as np
import pandas as pd
import pyprego

# Create a PSSM from a NumPy array
mat = np.array([
    [0.7, 0.1, 0.1, 0.1],
    [0.1, 0.1, 0.7, 0.1],
    [0.1, 0.1, 0.1, 0.7],
    [0.7, 0.1, 0.1, 0.1],
])
pssm = pyprego.pssm_dataframe(mat)
print(pssm)
#    pos    A    C    G    T
# 0    0  0.7  0.1  0.1  0.1
# 1    1  0.1  0.1  0.7  0.1
# 2    2  0.1  0.1  0.1  0.7
# 3    3  0.7  0.1  0.1  0.1
```

### Extracting the array

```python
arr = pyprego.pssm_to_array(pssm)
print(arr.shape)  # (4, 4)
```

## Consensus Sequence

Derive a consensus string from the PSSM:

```python
print(pyprego.consensus_from_pssm(pssm))  # "AGTA"
```

The consensus function uses IUPAC ambiguity codes when two nucleotides share a position. For example, a position with roughly equal A and G probabilities yields `R`. Positions with low information content are shown in lowercase.

Thresholds control the behaviour:

```python
# Stricter: require > 60% for a single-base call
consensus = pyprego.consensus_from_pssm(
    pssm,
    single_thresh=0.6,
    double_thresh=0.8,
)
```

## Information Content

The information content (in bits) at each position quantifies how far the nucleotide distribution deviates from uniform. Maximum is 2 bits (one nucleotide with probability 1); minimum is 0 bits (uniform).

```python
bits = pyprego.bits_per_pos(pssm)
print(bits)
# array([1.16, 1.16, 1.16, 1.16])  (approximate)
```

!!! note "Prior"
    The `prior` parameter (default 0.01) is added before computing information content. This avoids log(0) and matches the R implementation.

## Reverse Complement

```python
rc_pssm = pyprego.pssm_rc(pssm)
print(pyprego.consensus_from_pssm(rc_pssm))
# Reverse complement of "AGTA" -> "TACT"
```

`pssm_rc` reverses the row order and swaps A/T and C/G columns.

## Trimming

Remove low-information positions from the edges of a PSSM:

```python
# Create a PSSM with uninformative edges
mat_padded = np.vstack([
    np.full((3, 4), 0.25),  # uninformative prefix
    mat,
    np.full((3, 4), 0.25),  # uninformative suffix
])
pssm_padded = pyprego.pssm_dataframe(mat_padded)

trimmed = pyprego.pssm_trim(pssm_padded, bits_thresh=0.1)
print(f"Before: {len(pssm_padded)} positions")
print(f"After:  {len(trimmed)} positions")
```

The `bits_thresh` parameter sets the minimum information content (in bits) for a position to be kept.

## Adding a Prior

```python
pssm_prior = pyprego.pssm_add_prior(pssm, prior=0.05)
# Adds 0.05 to all values and re-normalizes each row to sum to 1
```

## Theoretical Score Bounds

```python
max_score = pyprego.pssm_theoretical_max(pssm)
min_score = pyprego.pssm_theoretical_min(pssm)
print(f"Score range: [{min_score:.3f}, {max_score:.3f}]")
```

These compute the maximum and minimum possible log-likelihood score over all possible sequences of the PSSM's length.

## Quantile Scores

Compute empirical score quantiles by scoring random sequences:

```python
q = pyprego.pssm_quantile(pssm, quantile=0.99, n_sequences=10000)
print(f"99th percentile score: {q:.3f}")
```

## Concatenating PSSMs

Join two PSSMs end-to-end, optionally with a gap:

```python
pssm2 = pyprego.pssm_dataframe(np.array([
    [0.1, 0.7, 0.1, 0.1],
    [0.1, 0.1, 0.7, 0.1],
]))

combined = pyprego.pssm_concat(pssm, pssm2, gap=2)
print(f"Length: {len(combined)}")
# Original (4) + gap (2, uniform) + second (2) = 8 positions
```

## Comparing PSSMs

### Correlation

Compute the best-alignment Pearson correlation between two PSSMs:

```python
r = pyprego.pssm_cor(pssm, rc_pssm)
print(f"Correlation: {r:.4f}")
```

The function slides one PSSM along the other and returns the maximum correlation across all alignments. Both orientations (forward and reverse complement) are tested.

### Difference

`pssm_diff` computes a distance metric between two PSSMs:

```python
d = pyprego.pssm_diff(pssm, rc_pssm)
print(f"Difference: {d:.4f}")
```

### Dataset-level comparison

Compare a PSSM against a collection of PSSMs (stored as a DataFrame with a name column):

```python
# Suppose `dataset` is a DataFrame of stacked PSSMs with a 'name' column
# pssm_dataset_cor returns a DataFrame of correlations per motif
cors = pyprego.pssm_dataset_cor(pssm, dataset)
print(cors.head())
```

## Matching Against Databases

`pssm_match` finds the best-matching motif in a database or dataset:

```python
# Against a MotifDB object
db = pyprego.create_motif_db(sequences)
match = pyprego.pssm_match(pssm, db)
print(match)

# Against a dataset DataFrame
datasets = pyprego.all_motif_datasets()
print(datasets)  # Available bundled datasets
```

!!! warning "Database format"
    `pssm_match` accepts either a `MotifDB` object or a dataset DataFrame.
    The bundled datasets can be accessed via `all_motif_datasets()`.

## Summary of PSSM Functions

| Function | Description |
|----------|-------------|
| `pssm_dataframe(array)` | Create PSSM DataFrame from (L, 4) array |
| `pssm_to_array(pssm)` | Extract (L, 4) array from PSSM DataFrame |
| `consensus_from_pssm(pssm)` | Consensus string with IUPAC codes |
| `bits_per_pos(pssm)` | Information content per position (bits) |
| `pssm_rc(pssm)` | Reverse complement |
| `pssm_trim(pssm)` | Trim low-information edges |
| `pssm_add_prior(pssm, prior)` | Add prior and re-normalize |
| `pssm_theoretical_max(pssm)` | Maximum possible score |
| `pssm_theoretical_min(pssm)` | Minimum possible score |
| `pssm_quantile(pssm, q)` | Empirical score quantile |
| `pssm_concat(*pssms, gap)` | Concatenate PSSMs with optional gap |
| `pssm_cor(p1, p2)` | Best-alignment correlation |
| `pssm_diff(p1, p2)` | Best-alignment distance |
| `pssm_dataset_cor(pssm, dataset)` | Correlation against a motif dataset |
| `pssm_dataset_diff(pssm, dataset)` | Distance against a motif dataset |
| `pssm_match(pssm, db)` | Best match in a database |
