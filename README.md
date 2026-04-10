# ITSMixer: Iterative Time-Mixing MLPs for Efficient Long-Term Forecasting

Official implementation of the paper:

**ITSMixer: iterative time-mixing MLPs for efficient long-term forecasting**  
**Authors:** Arian Lotfi and Siavash Damari  
**Journal:** *Evolving Systems*  
**DOI:** 10.1007/s12530-026-09830-0  
**Paper:** https://link.springer.com/article/10.1007/s12530-026-09830-0

---

## Overview

ITSMixer is a lightweight MLP-based model for multivariate long-term time series forecasting. Building on TSMixer, it removes feature-mixing layers and relies on iterative time-mixing MLPs for efficient temporal modeling.

The model achieves competitive performance with reduced computational overhead, making it suitable for real-world applications where efficiency and simplicity are critical.

---

## Repository Contents

- `ITSMixer.py` — core implementation of the ITSMixer model  
- `Experiments/` — experiment configurations and logs  

---

## Model Components

Key components in `ITSMixer.py` include:

- **RevIN** — reversible instance normalization  
- **Mlp_time** — time-mixing MLP with GELU activation and dropout  
- **Mixer_Layer** — iterative residual chaining with multiple MLPs  
- **Backbone** — mixer with temporal projection  
- **Model** — full wrapper with normalization and denormalization  

---

## Status

This repository is now public. Additional documentation, licensing information, and extended code details will be added in future updates.

---

## Citation

If you use this work, please cite:

```bibtex
@article{lotfi2026itsmixer,
  title   = {ITSMixer: iterative time-mixing MLPs for efficient long-term forecasting},
  author  = {Lotfi, Arian and Damari, Siavash},
  journal = {Evolving Systems},
  year    = {2026},
  doi     = {10.1007/s12530-026-09830-0},
  url     = {https://link.springer.com/article/10.1007/s12530-026-09830-0}
}
