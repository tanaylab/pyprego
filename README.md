# pyprego

Python implementation of the [prego](https://github.com/tanaylab/prego) R package — a PWM Regression Optimizer for motif discovery in DNA sequences.

## Installation

```bash
pip install -e .
```

Optional dependencies:
```bash
pip install pymisha   # for genomic interval integration
pip install logomaker  # for sequence logo plots
```

## Quick Start

### Continuous regression (find motifs correlated with a response)

```python
import pyprego

# sequences: list of equal-length DNA strings
# response: 1D or 2D numpy array (one row per sequence)
result = pyprego.regress_pwm(sequences, response)

# Result contains:
result.pssm       # PSSM DataFrame (pos, A, C, G, T)
result.spat       # Spatial model DataFrame (bin, spat_factor)
result.pred       # Predictions for each sequence
result.consensus  # Consensus motif string
result.r2         # R-squared per response dimension

# Predict on new sequences
new_scores = result.predict(new_sequences)
```

### Binary classification (find motifs that discriminate two classes)

```python
result = pyprego.regress_pwm(
    sequences, binary_response,  # 0/1 vector
    score_metric="ks"
)
result.ks    # KS test statistic
result.pred  # Predictions
```

### Multiple motifs

```python
result = pyprego.regress_pwm(sequences, response, motif_num=3)
result.models      # List of individual motif models
result.multi_stats # Statistics for each motif
result.pred        # Combined predictions
```

### PWM scoring with known motif

```python
scores = pyprego.compute_pwm(sequences, pssm, spat=spat_model, bidirect=True)
local_scores = pyprego.compute_local_pwm(sequences, pssm)
```

### K-mer screening

```python
kmers = pyprego.screen_kmers(sequences, response, kmer_len=8)
print(kmers.head())  # Top correlated k-mers
```

### PSSM utilities

```python
pyprego.pssm_cor(pssm1, pssm2)       # Correlation between PSSMs
pyprego.pssm_match(pssm, motif_db)   # Match against database
pyprego.bits_per_pos(pssm)            # Information content
pyprego.consensus_from_pssm(pssm)     # Consensus sequence
pyprego.pssm_rc(pssm)                 # Reverse complement
pyprego.pssm_trim(pssm)              # Trim low-info edges
```

### Model export/import

```python
from pyprego.export import export_regression_model, load_regression_model

export_regression_model(result, "model.json")
loaded = load_regression_model("model.json")
new_scores = loaded.predict(new_sequences)
```

## API Compatibility with R prego

pyprego implements the same functions as the R package:

| R function | Python function | Status |
|---|---|---|
| `regress_pwm()` | `pyprego.regress_pwm()` | Complete |
| `regress_multiple_motifs()` | `pyprego.regress_pwm(motif_num=N)` | Complete |
| `compute_pwm()` | `pyprego.compute_pwm()` | Complete |
| `compute_local_pwm()` | `pyprego.compute_local_pwm()` | Complete |
| `screen_kmers()` | `pyprego.screen_kmers()` | Complete |
| `generate_kmers()` | `pyprego.generate_kmers()` | Complete |
| `kmer_matrix()` | `pyprego.kmer_matrix()` | Complete |
| `pssm_cor()` / `pssm_diff()` | `pyprego.pssm_cor()` / `pyprego.pssm_diff()` | Complete |
| `pssm_match()` | `pyprego.pssm_match()` | Complete |
| `pssm_trim()` / `pssm_rc()` | `pyprego.pssm_trim()` / `pyprego.pssm_rc()` | Complete |
| `bits_per_pos()` | `pyprego.bits_per_pos()` | Complete |
| `create_motif_db()` | `pyprego.create_motif_db()` | Complete |
| `extract_pwm()` | `pyprego.motif_db.extract_pwm()` | Complete |
| `plot_pssm_logo()` | `pyprego.plot_pssm_logo()` | Complete |
| `intervals_to_seq()` | `pyprego.intervals_to_seq()` | Complete (requires pymisha) |
| `gextract_pwm()` | `pyprego.gextract_pwm()` | Complete (requires pymisha) |

## Testing

```bash
# Fast tests (~6 seconds)
pytest tests/ --ignore=tests/test_high_level.py --ignore=tests/test_regression.py --ignore=tests/test_integration.py

# Full suite (includes slow regression tests)
pytest tests/
```

## Architecture

- **NumPy-based**: All computation uses NumPy arrays (no GPU/PyTorch dependency)
- **pandas DataFrames**: PSSMs and spatial models use DataFrames matching R conventions
- **Optional pymisha**: Genomic functions work when pymisha is installed
- **GPU-ready design**: Clean array interfaces allow future torch tensor swap

See [DECISIONS.md](DECISIONS.md) for detailed architecture decisions.

## License

MIT
