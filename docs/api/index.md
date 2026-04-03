# API Reference

This section documents the full public API of pyprego. Functions are organized by module.

## Modules

| Module | Description |
|--------|-------------|
| [Regression](regression.md) | Motif discovery via iterative PWM regression |
| [PWM Scoring](compute.md) | Score sequences against a known PSSM |
| [K-mer Operations](kmers.md) | K-mer generation, counting, and screening |
| [PSSM Utilities](pssm.md) | PSSM manipulation, comparison, and analysis |
| [Motif Database](motif-db.md) | Motif database management and querying |
| [Visualization](visualization.md) | Sequence logos and regression diagnostics |
| [Genomic](genomic.md) | Genomic interval integration (requires pymisha) |
| [Export/Import](export.md) | Model serialization and deserialization |
| [Types](types.md) | Core data structures and type definitions |

## Quick Import

All public functions are available directly from the top-level `pyprego` namespace:

```python
import pyprego

result = pyprego.regress_pwm(sequences, response)
scores = pyprego.compute_pwm(sequences, pssm)
kmers  = pyprego.screen_kmers(sequences, response, kmer_len=8)
```

Alternatively, import from submodules:

```python
from pyprego.regression import regress_pwm
from pyprego.compute import compute_pwm
from pyprego.kmers import screen_kmers
```
