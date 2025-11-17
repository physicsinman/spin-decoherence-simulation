# Legacy Files

This directory contains backup copies of original root-level files that have been converted to compatibility wrappers.

## What Changed

The root-level files (`ornstein_uhlenbeck.py`, `config.py`, `units.py`, `noise_models.py`, `coherence.py`, `fitting.py`) have been converted to compatibility wrappers that re-export functions from the `spin_decoherence` package.

## Migration Guide

### Old Import (Still Works)
```python
from ornstein_uhlenbeck import generate_ou_noise
from config import CONSTANTS
from units import Units
from coherence import compute_ensemble_coherence
from fitting import fit_coherence_decay
```

### New Import (Recommended)
```python
from spin_decoherence.noise import generate_ou_noise
from spin_decoherence.config import CONSTANTS, Units
from spin_decoherence.physics import compute_ensemble_coherence
from spin_decoherence.analysis import fit_coherence_decay
```

## Benefits

1. **No Breaking Changes**: Existing code continues to work
2. **Single Source of Truth**: All functionality is in `spin_decoherence` package
3. **Easier Maintenance**: Only one version of each function to maintain
4. **Better Organization**: Clear package structure

## Files Converted

- `ornstein_uhlenbeck.py` → wrapper for `spin_decoherence.noise`
- `config.py` → wrapper for `spin_decoherence.config`
- `units.py` → wrapper for `spin_decoherence.config`
- `noise_models.py` → wrapper for `spin_decoherence.noise`
- `coherence.py` → wrapper for `spin_decoherence.physics`
- `fitting.py` → wrapper for `spin_decoherence.analysis`

## Date

November 2024

