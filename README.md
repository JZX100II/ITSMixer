# ITSMixer: Iterative Time-Mixing MLPs for Efficient Long-Term Forecasting

Official partial code release for the paper "[TSMixer: Iterative Time-Mixing MLPs for Efficient Long-Term Forecasting]".

This repository provides:
- The core model architecture (`ITSMixer.py`).
- Experiment-specific configurations and logs (in `Experiments/`).

## Model Overview
ITSMixer is a lightweight MLP-based model for multivariate long-term time series forecasting. It features an iterative time-mixing mechanism with residual accumulation across multiple MLP instances, enabling efficient temporal mixing without attention or recurrence overhead.

Key components in `ITSMixer.py`:
- RevIN (reversible instance normalization)
- Mlp_time (time-mixing MLP with GELU and dropout)
- Mixer_Layer (iterative residual chaining with multiple MLPs and loops)
- Backbone (mixer + temporal projection)
- Model (full wrapper with RevIN norm/denorm)

## Code availability
This repository contains the core implementation of ITSMixer for validation purposes during the review process.

The code remains unpublished and is not licensed for reuse, redistribution, modification, or any other purpose at this time (copyright © Arian Lotfi 2026).  
A full open-source release with an appropriate license is planned upon acceptance and publication of the paper.