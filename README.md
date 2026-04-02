# ITSMixer: Iterative Time-Mixing MLPs for Efficient Long-Term Forecasting

Official implementation of the paper:

**ITSMixer: iterative time-mixing MLPs for efficient long-term forecasting**  
**Authors:** Arian Lotfi, Siavash Damari  
**Journal:** *Evolving Systems*  
**DOI:** 10.1007/s12530-026-09830-0

## Overview

ITSMixer is a lightweight MLP-based model for multivariate long-term time series forecasting. Building on TSMixer, it removes feature-mixing layers and relies on iterative time-mixing MLPs for efficient temporal modeling.

The model is designed to provide strong forecasting performance with lower computational complexity, making it a practical choice for real-world applications where efficiency and simplicity are important.

## Repository Contents

This repository currently includes:

- `ITSMixer.py` — core implementation of the ITSMixer model
- `Experiments/` — experiment configurations and logs

## Model Components

Key components in `ITSMixer.py` include:

- **RevIN** — reversible instance normalization
- **Mlp_time** — time-mixing MLP with GELU activation and dropout
- **Mixer_Layer** — iterative residual chaining with multiple MLPs and loops
- **Backbone** — mixer with temporal projection
- **Model** — full wrapper with normalization and denormalization

## Status

This repository is now public. Additional documentation, licensing information, and extended code details will be added in future updates.

## Citation

If you use this work, please cite:

```bibtex
@article{lotfi2026itsmixer,
  title   = {ITSMixer: iterative time-mixing MLPs for efficient long-term forecasting},
  author  = {Lotfi, Arian and Damari, Siavash},
  journal = {Evolving Systems},
  year    = {2026},
  doi     = {10.1007/s12530-026-09830-0}
}
