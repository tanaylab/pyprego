# pyprego

**PWM Regression Optimizer for motif discovery in DNA sequences.**

pyprego is a Python port of the [prego](https://github.com/tanaylab/prego) R package. It discovers sequence motifs (position weight matrices) that best explain a quantitative or binary response variable across a set of DNA sequences, using iterative coordinate-descent optimization of PSSM weights and a spatial positional model.

## Key Features

- **Motif discovery** -- `regress_pwm` iteratively optimizes a PSSM and spatial model to maximize correlation (or KS statistic) with a response variable.
- **K-mer screening** -- fast, vectorized k-mer frequency correlation to identify candidate seed motifs, including gapped k-mers.
- **PWM scoring** -- `compute_pwm` scores arbitrary sequences against any PSSM with spatial weighting (logSumExp or max aggregation).
- **Multiple motifs** -- sequential residual-based regression for discovering more than one motif per dataset.
- **Motif database** -- bundled JASPAR and HOMER databases for motif matching and enrichment analysis.
- **Visualization** -- sequence logos, spatial model plots, and regression QC diagnostics.
- **Optional C extension** -- compiled accelerators for energy computation and k-mer counting when available.
- **Genomic integration** -- extract sequences from genomic intervals via [pymisha](https://github.com/tanaylab/pymisha) and run PWM regression directly on genomic data.

## Installation

```bash
pip install pyprego
```

## Quick Example

```python
import numpy as np
import pyprego

# Generate example data: 500 sequences of length 200
rng = np.random.default_rng(42)
seqs = ["".join(rng.choice(list("ACGT"), 200)) for _ in range(500)]
response = rng.normal(size=500)

# Discover a motif
result = pyprego.regress_pwm(seqs, response)
print(result.consensus)   # e.g. "TGANNTCA"
print(result.r2)          # R-squared of the fit
```

## What Next?

- [Getting Started](guides/quickstart.md) -- installation and first steps.
- [Motif Discovery Tutorial](tutorials/motif-discovery.md) -- end-to-end walkthrough of PWM regression.
- [API Reference](api/index.md) -- full function documentation.
