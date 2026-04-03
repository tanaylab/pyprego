# Motif Discovery with PWM Regression

This tutorial walks through the complete workflow for discovering DNA sequence motifs using pyprego's iterative PWM regression.

## What is PWM Regression?

Position Weight Matrix (PWM) regression finds a motif -- represented as a PSSM (Position-Specific Scoring Matrix) -- whose binding score across input sequences best correlates with a response variable. The algorithm:

1. **Seeds** from a k-mer that correlates with the response.
2. **Iteratively optimizes** PSSM weights and a spatial positional model via coordinate descent.
3. **Returns** the optimized PSSM, spatial model, and per-sequence predictions.

The spatial model captures where in the sequence the motif tends to appear. Together, the PSSM and spatial model define a generative scoring function: for each sequence, the score aggregates (via logSumExp) the log-likelihood of the motif at every position, weighted by the spatial factor at that position.

## Generating Test Data

For this tutorial we will embed a known motif into random sequences so we can verify recovery.

```python
import numpy as np
import pyprego

rng = np.random.default_rng(42)

n_seqs = 1000
seq_len = 200
motif = "GATAAG"
motif_len = len(motif)

# Generate random background sequences
seqs = []
response = np.zeros(n_seqs)
for i in range(n_seqs):
    seq = list(rng.choice(list("ACGT"), seq_len))
    # Embed the motif in ~50% of sequences at a random position
    if rng.random() < 0.5:
        pos = rng.integers(0, seq_len - motif_len)
        for j, base in enumerate(motif):
            seq[pos + j] = base
        response[i] = 1.0
    seqs.append("".join(seq))
```

## Running regress_pwm

### Basic invocation

```python
result = pyprego.regress_pwm(seqs, response)

print(f"Consensus: {result.consensus}")
print(f"R-squared: {result.r2:.4f}")
```

The result should recover something close to `GATAAG` (or its reverse complement `CTTATC`).

### Key parameters

| Parameter       | Default | Description |
|-----------------|---------|-------------|
| `motif_length`  | 15      | Length of the PSSM to optimize |
| `score_metric`  | `"r2"`  | `"r2"` for continuous, `"ks"` for binary response |
| `bidirect`      | `True`  | Score both DNA strands |
| `spat_bin_size` | `None`  | Spatial bin size (auto-calculated if `None`) |
| `multi_kmers`   | `False` | Try multiple k-mer seeds and pick the best |
| `kmer_length`   | 8       | Length of k-mers to screen for seeding |
| `verbose`       | `False` | Print optimization progress |

### Using a specific seed motif

You can bypass k-mer screening by providing a seed motif directly:

```python
result = pyprego.regress_pwm(
    seqs, response,
    motif="GATAAG",    # start from this k-mer
    motif_length=10,   # optimize a 10-bp PSSM
)
```

You can also pass a PSSM DataFrame as the `motif` argument to start from an existing weight matrix.

### Binary vs continuous response

For binary (0/1) response, use the KS metric:

```python
binary_response = (response > 0).astype(float)

result_ks = pyprego.regress_pwm(
    seqs, binary_response,
    score_metric="ks",
)
print(f"KS statistic: {result_ks.ks:.4f}")
```

## Interpreting Results

### The PSSM

The PSSM is a pandas DataFrame with columns `pos`, `A`, `C`, `G`, `T`:

```python
print(result.pssm)
#    pos         A         C         G         T
# 0    0  0.250000  0.250000  0.250000  0.250000
# 1    1  0.012345  0.012345  0.962963  0.012345
# 2    2  0.962963  0.012345  0.012345  0.012345
# ...
```

Each row is a position in the motif. Values are (unnormalized) weights; higher values indicate stronger preference for that nucleotide at that position. Edge positions with near-uniform weights carry little information.

### The spatial model

The spatial model describes the positional bias of the motif along the sequence:

```python
print(result.spat)
#    bin  spat_factor
# 0    0     1.023456
# 1    1     0.987654
# ...
```

A flat spatial model (all factors near 1.0) means the motif can appear anywhere. Non-uniform factors indicate positional preferences.

### Predictions

```python
# Per-sequence predicted scores
print(result.pred[:5])

# Score new sequences
new_scores = result.predict(new_sequences)
```

### Consensus sequence

```python
print(result.consensus)  # e.g. "GATAAG"
```

The consensus uses IUPAC codes when two nucleotides share a position (e.g. `R` for A/G, `Y` for C/T). Positions with low information content appear as lowercase.

## Multi-Kmer Mode

When the true motif is unknown, testing multiple k-mer seeds can improve results:

```python
result = pyprego.regress_pwm(
    seqs, response,
    multi_kmers=True,
    max_cands=10,        # try top 10 k-mer seeds
    kmer_length=8,
    max_gap=1,           # also try gapped k-mers
    val_frac=0.1,        # hold out 10% for validation
    verbose=True,
)
```

This screens k-mers, runs regression from each candidate seed, and selects the model with the best validation score.

## Multiple Motif Discovery

When more than one motif drives the response, use `regress_multiple_motifs`:

```python
multi = pyprego.regress_multiple_motifs(
    seqs, response,
    motif_num=3,
    verbose=True,
)

# Individual motif models
for i, model in enumerate(multi.models):
    print(f"Motif {i+1}: {model.consensus}")

# Statistics table
print(multi.multi_stats)
#    model     score  comb_score      diff consensus seed_motif
# 0      1  0.312345    0.312345       NaN   GATAAG   GATAAGNC
# 1      2  0.045678    0.356789  0.044444   CACGTG   CACGTGNC
# 2      3  0.012345    0.367890  0.011101   TGANTCA  TGANTCAN

# Combined prediction
print(f"Combined R2: {multi.multi_stats['comb_score'].iloc[-1]:.4f}")

# Predict on new sequences
combined_scores = multi.predict(new_sequences)
```

The algorithm works by regressing the first motif, computing smoothed residuals, and regressing subsequent motifs on those residuals. A combined linear model is fit at each step.

## Using the Motif Database

pyprego ships with bundled motif databases (JASPAR, HOMER). You can match discovered motifs against known transcription factor binding sites:

```python
# List available databases
print(pyprego.all_motif_datasets())

# Match against the default database
result = pyprego.regress_pwm(
    seqs, response,
    match_with_db=True,
)

print(f"Best DB match: {result.db_match_motif}")
print(f"Correlation:   {result.db_match_cor:.3f}")
```

### Manual database matching

```python
# Create a motif database from a dataset
db = pyprego.create_motif_db(seqs)

# Match a PSSM against it
match_result = pyprego.pssm_match(result.pssm, db)
print(match_result)

# Get the PSSM for a specific motif
motif_pssm = pyprego.get_motif_pssm("JASPAR", "MA0002.1")
```

### Motif enrichment

Screen all motifs in a database for enrichment relative to a response:

```python
enrichment = pyprego.motif_enrichment(seqs, response, db)
print(enrichment.head())
```

## Visualization

### Sequence logo

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 2))
pyprego.plot_pssm_logo(result.pssm, ax=ax)
ax.set_title(f"Discovered motif: {result.consensus}")
plt.tight_layout()
plt.show()
```

!!! tip "logomaker"
    If `logomaker` is installed, `plot_pssm_logo` produces publication-quality
    letter logos. Otherwise it falls back to stacked bar charts, which still
    convey the same information content.

### Spatial model

```python
fig, ax = plt.subplots(figsize=(6, 3))
pyprego.plot_spat_model(result.spat, ax=ax)
plt.tight_layout()
plt.show()
```

### Regression QC

```python
# For continuous response
fig = pyprego.plot_regression_qc(result, response)

# For binary response
fig = pyprego.plot_regression_qc(result_ks, binary_response)
```

The QC plot shows prediction vs response (scatter for continuous, box plots for binary) along with the PSSM logo and spatial model.

## Exporting and Loading Models

### Single model

```python
# Export
pyprego.export_regression_model(result, "my_motif.json")

# Load
loaded = pyprego.load_regression_model("my_motif.json")
scores = loaded.predict(seqs)
```

### Multi-motif model

```python
pyprego.export_multi_regression(multi, "multi_motifs.json")
loaded_multi = pyprego.load_multi_regression("multi_motifs.json")
```

The JSON format stores the PSSM, spatial model, and all metadata needed to reconstruct the scoring function. Predictions from a loaded model are numerically identical to the original.

## Advanced Options

### Optimizer control

```python
result = pyprego.regress_pwm(
    seqs, response,
    resolutions=[0.5, 0.1, 0.05, 0.01],  # PSSM optimization step sizes
    spat_resolutions=[0.2, 0.05, 0.01],   # spatial model step sizes
    improve_epsilon=1e-5,   # convergence threshold
    min_nuc_prob=0.001,     # minimum nucleotide probability
    unif_prior=0.05,        # uniform prior added to seed PSSM
)
```

The optimizer runs through each resolution from coarse to fine. At each resolution it tries all 20 perturbation moves at every PSSM position and accepts the move that most improves the score, stopping when no move improves by more than `improve_epsilon`.

### Spatial model options

```python
result = pyprego.regress_pwm(
    seqs, response,
    spat_bin_size=10,        # 10-bp spatial bins
    spat_num_bins=20,        # 20 bins total
    symmetrize_spat=True,    # force symmetric spatial model
    optimize_spat=True,      # optimize spatial model (default)
)
```

### Cross-validation

```python
cv_result = pyprego.regress_pwm_cv(
    seqs, response,
    n_folds=5,
)

print(f"Mean CV R2: {cv_result.mean_score:.4f}")
print(f"Fold scores: {cv_result.fold_scores}")
```

### Cluster-specific regression

When sequences come from different clusters (e.g. cell types), regress separately per cluster:

```python
clusters = np.array(["A", "A", "B", "B", ...])

cluster_result = pyprego.regress_pwm_clusters(
    seqs, response, clusters,
)

for name, model in cluster_result.models.items():
    print(f"Cluster {name}: {model.consensus} (R2={model.r2:.3f})")
```
