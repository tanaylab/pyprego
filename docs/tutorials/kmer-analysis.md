# K-mer Analysis

This tutorial covers k-mer screening and the k-mer utilities in pyprego, which are useful both as a standalone analysis tool and as a precursor to full PWM regression.

## K-mer Screening Basics

`screen_kmers` computes the Pearson correlation between each k-mer's occurrence count (across sequences) and a response variable. This provides a quick, interpretable view of which short sequence features drive the response.

```python
import numpy as np
import pyprego

# Example data
rng = np.random.default_rng(42)
seqs = ["".join(rng.choice(list("ACGT"), 200)) for _ in range(500)]
response = rng.normal(size=500)

# Screen all 8-mers
kmers_df = pyprego.screen_kmers(seqs, response, kmer_len=8)
print(kmers_df.head(10))
```

The result is a DataFrame with columns:

| Column   | Description |
|----------|-------------|
| `kmer`   | The k-mer string |
| `max_r2` | Maximum R-squared across response columns |
| `avg_n`  | Mean count of the k-mer per sequence |
| `avg_var`| Variance of the count across sequences |
| `r0`     | Pearson correlation with the response (one column per response dimension) |

Rows are sorted by `max_r2` in descending order.

### Filtering by correlation

Use `min_cor` to keep only k-mers above a correlation threshold:

```python
kmers_df = pyprego.screen_kmers(
    seqs, response,
    kmer_len=8,
    min_cor=0.05,  # only keep |r| >= 0.05
)
```

!!! note "Interpretation"
    `min_cor` filters on the absolute Pearson correlation coefficient, not R-squared. A k-mer passes if any of its response-column correlations has `|r| >= min_cor`.

### Multi-dimensional response

When the response has multiple columns, `screen_kmers` computes correlations against each column independently:

```python
response_2d = rng.normal(size=(500, 3))
kmers_df = pyprego.screen_kmers(seqs, response_2d, kmer_len=6)
print(kmers_df.columns)
# ['kmer', 'max_r2', 'avg_n', 'avg_var', 'r0', 'r1', 'r2']
```

## Gapped K-mers

Gapped k-mers contain wildcard positions (`N`) that match any nucleotide. They capture motifs with variable spacers -- common in transcription factor binding sites (e.g. nuclear receptor half-sites separated by a variable gap).

```python
kmers_df = pyprego.screen_kmers(
    seqs, response,
    kmer_len=8,
    max_gap=2,   # allow up to 2 contiguous N positions
    min_gap=1,   # require at least 1 gap position
)

# Gapped k-mers look like: "ACNNNGTC", "GATNCCAT", etc.
print(kmers_df[kmers_df["kmer"].str.contains("N")].head())
```

### How gaps work

The `generate_kmers` function produces all k-mers where a contiguous block of 1 to `max_gap` positions is replaced with `N`:

```python
# Standard 4-mers: 256 total
standard = pyprego.generate_kmers(4)
print(len(standard))  # 256

# With gaps: adds gapped variants
gapped = pyprego.generate_kmers(4, max_gap=1)
print(len(gapped))  # 256 + gapped variants

# See some gapped k-mers
print([k for k in gapped if "N" in k][:10])
```

When counting occurrences in `kmer_matrix`, `N` positions match any nucleotide, so `"ACNGT"` matches `"ACAGT"`, `"ACCGT"`, `"ACGGT"`, and `"ACTGT"`.

## K-mer Count Matrix

For custom analysis, build the full k-mer count matrix:

```python
kmer_list = pyprego.generate_kmers(6)
km_matrix = pyprego.kmer_matrix(seqs, kmer_list)

print(km_matrix.shape)  # (500, 4096) for 6-mers
print(km_matrix.iloc[:5, :5])
```

The returned DataFrame has one row per sequence and one column per k-mer, with integer counts.

!!! tip "Memory"
    For long k-mers, the number of columns grows as `4^k`. An 8-mer matrix has 65,536 columns. Consider using gapped k-mers or shorter k-mers if memory is a concern.

### Custom screening with the count matrix

```python
from scipy.stats import spearmanr

# Build count matrix once
km = pyprego.kmer_matrix(seqs, pyprego.generate_kmers(6))

# Spearman correlation instead of Pearson
correlations = {}
for col in km.columns:
    r, p = spearmanr(km[col].values, response)
    correlations[col] = (r, p)

# Find top k-mers
top = sorted(correlations.items(), key=lambda x: abs(x[1][0]), reverse=True)[:10]
for kmer, (r, p) in top:
    print(f"{kmer}: rho={r:.4f}, p={p:.2e}")
```

## Converting K-mers to PSSMs

A k-mer can be converted to a PSSM for use with `compute_pwm` or as a seed for regression:

```python
pssm = pyprego.kmers_to_pssm("GATAAG")
print(pssm)
#    pos     A      C      G      T
# 0    0  0.01  0.01  0.97  0.01
# 1    1  0.97  0.01  0.01  0.01
# 2    2  0.01  0.01  0.01  0.97
# 3    3  0.97  0.01  0.01  0.01
# 4    4  0.97  0.01  0.01  0.01
# 5    5  0.01  0.01  0.97  0.01
```

For gapped k-mers, `N` positions get uniform weights (0.25 each):

```python
pssm_gapped = pyprego.kmers_to_pssm("GATNAG")
print(pssm_gapped.iloc[3])
# pos    3.00
# A      0.25
# C      0.25
# G      0.25
# T      0.25
```

### Multiple k-mers

```python
pssm_multi = pyprego.kmers_to_pssm(["GATAAG", "CACGTG"])
print(pssm_multi)  # Combined DataFrame with a 'kmer' column
```

## Converting PSSMs to K-mers

The reverse operation extracts the top-scoring k-mer from a PSSM:

```python
kmer = pyprego.pssm_to_kmer(result.pssm)
print(kmer)  # e.g. "GATAAG"
```

## Integration with Regression

K-mer screening is used internally by `regress_pwm` when no seed motif is provided. You can replicate this workflow explicitly:

```python
# Step 1: Screen k-mers
kmers_df = pyprego.screen_kmers(seqs, response, kmer_len=8, max_gap=1)
top_kmer = kmers_df.iloc[0]["kmer"]
print(f"Best seed: {top_kmer}")

# Step 2: Use top k-mer as seed for regression
result = pyprego.regress_pwm(seqs, response, motif=top_kmer)
print(f"Consensus: {result.consensus}, R2: {result.r2:.4f}")
```

### Trying multiple seeds manually

```python
results = []
for _, row in kmers_df.head(5).iterrows():
    res = pyprego.regress_pwm(seqs, response, motif=row["kmer"])
    results.append((row["kmer"], res.r2, res.consensus))
    print(f"Seed: {row['kmer']}  ->  {res.consensus}  (R2={res.r2:.4f})")

# Pick the best
best = max(results, key=lambda x: x[1])
print(f"\nBest: seed={best[0]}, consensus={best[2]}, R2={best[1]:.4f}")
```

This is essentially what `regress_pwm(multi_kmers=True)` does automatically, with the addition of a validation holdout for model selection.
