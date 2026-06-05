# ITSMixer: Iterative Time-Mixing MLPs for Efficient Long-Term Forecasting

**Official implementation** of the paper:

**ITSMixer: iterative time-mixing MLPs for efficient long-term forecasting**

**Authors:** Arian Lotfi and Siavash Damari

**Journal:** *Evolving Systems*

**DOI:** 10.1007/s12530-026-09830-0

**Paper:** [https://link.springer.com/article/10.1007/s12530-026-09830-0](https://link.springer.com/article/10.1007/s12530-026-09830-0)

---

## Overview

ITSMixer is a lightweight, pure MLP-based model for multivariate long-term time series forecasting. It builds upon the TSMixer architecture by removing feature-mixing layers and introducing **iterative time-mixing MLPs** for effective temporal pattern capture.

The model delivers competitive forecasting performance with comparable computational overhead, making it ideal for real-world applications that prioritize efficiency and simplicity.

### Key Advantages

- **Efficiency**: Pure MLP design — no attention or convolutions
- **Performance**: Strong results on standard benchmarks
- **Scalability**: Low memory usage and fast inference
- **Simplicity**: Easy to understand, modify, and deploy

---

## Quick Start

### 1. Clone the benchmark framework

```bash
git clone https://github.com/hughxx/tsf-new-paper-taste.git
cd tsf-new-paper-taste
```

### 2. Add ITSMixer files

```bash
# Copy model and experiment handler
cp /path/to/ITSMixer/ITSMixer.py models/
cp /path/to/ITSMixer/exp_main_itsmixer.py exp/exp_main.py
```

### 3. Run training

```bash
# Example: ETTh1 dataset (as provided)
python run.py --model ITSMixer --data_path ETTh1.csv --data ETTh1 --seq_len 512 --pred_len 96 --train_epochs 35
```

**Other common configurations:**

```bash
# Short-term (96 steps)
python run.py --model ITSMixer --data ETTh1 --seq_len 512 --pred_len 96 --train_epochs 20

# Long-term (720 steps)
python run.py --model ITSMixer --data ETTh1 --seq_len 512 --pred_len 720 --train_epochs 15
```

---

## Installation

### Prerequisites

- Python 3.8+
- PyTorch
- NumPy, Pandas
- scikit-learn, Matplotlib

**Recommended:** Use the Colab notebook style setup shown in the original repository or the quick start above.

---

## Using ITSMixer in Your Own Code

```python
import torch
from ITSMixer import Model

# Configuration
class Config:
    seq_len = 96   # Input sequence length
    pred_len = 24  # Prediction horizon
    enc_in = 7     # Number of input features

config = Config()
model = Model(config)

# Set device (uses GPU if available, otherwise falls back to CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Prepare dummy inputs
batch_x = torch.randn(32, config.seq_len, config.enc_in).to(device)
batch_x_mark = None
dec_inp = torch.randn(32, config.pred_len, config.enc_in).to(device)
batch_y_mark = None

# Forward pass
output = model(batch_x, batch_x_mark, dec_inp, batch_y_mark) 

print("Output shape:", output.shape)
```

---

## Model Architecture

ITSMixer consists of:

- **RevIN**: Reversible Instance Normalization (normalization + denormalization)
- **Mlp_time**: Time-mixing MLP (3 FC layers + GELU + dropout)
- **Mixer_Layer**: Iterative residual chaining of multiple time-mixing MLPs
- **Temporal Projection**: Projects from `seq_len` to `pred_len`

### Core Design Choices

- Iterative residual connections for deep yet stable training
- Pure MLP architecture (no attention/conv) → faster and lighter
- Strong emphasis on temporal mixing

**Architecture Flow** (simplified):

```
Input (B, T, C)
    ↓
RevIN Normalization
    ↓
Iterative Time-Mixing MLPs (with residuals)
    ↓
Temporal Projection (T → pred_len)
    ↓
RevIN Denormalization
    ↓
Output (B, pred_len, C)
```

---

## Supported Datasets

ITSMixer was evaluated on standard multivariate long-term forecasting benchmarks:

**ETT** (Electricity Transformer Temperature)
- ETTh1 / ETTh2
- ETTm1 / ETTm2

**Others**:
- Weather
- Electricity

For full dataset details, see the [tsf-new-paper-taste repository](https://github.com/hughxx/tsf-new-paper-taste).

---

## Results

ITSMixer achieves competitive or state-of-the-art performance with a much simpler architecture. Detailed results and comparisons are available in the [paper](https://link.springer.com/article/10.1007/s12530-026-09830-0).

### Selected Results (MSE / MAE)

**ETTh1 (hourly)**

| T (pred_len) | ITSMixer          | TSMixer          | PatchTST         |
|--------------|-------------------|------------------|------------------|
| 96           | **0.348 / 0.384** | 0.361 / 0.392    | 0.370 / 0.400    |
| 192          | **0.383 / 0.408** | 0.404 / 0.418    | 0.413 / 0.429    |
| 336          | **0.376 / 0.412** | 0.420 / 0.431    | 0.422 / 0.440    |
| 720          | **0.407 / 0.439** | 0.472 / 0.492    | 0.447 / 0.468    |

**ETTh2 (hourly)**

| T (pred_len) | ITSMixer            | TSMixer          | PatchTST         |
|--------------|---------------------|------------------|------------------|
| 96           | **0.259 / 0.327**   | 0.274 / 0.341    | 0.274 / 0.337    |
| 192          | **0.302 / 0.363**   | 0.339 / 0.385    | 0.341 / 0.382    |
| 336          | **0.314 / 0.378**   | 0.361 / 0.406    | 0.329 / 0.384    |
| 720          | 0.381 / 0.426       | 0.445 / 0.470    | **0.379 / 0.422**|

ITSMixer consistently ranks among the top performers (often best) across datasets and horizons while using a lightweight pure-MLP design.

---

## Repository Contents

- **`ITSMixer.py`** — Core model implementation
- **`exp_main_itsmixer.py`** — Modified experiment handler for integration
- **`Experiments/`** — Experiment configs and logs
- `LICENSE`, `README.md`

---

## Integration with tsf-new-paper-taste Framework

This repo provides the minimal files needed to integrate ITSMixer into the unofficial benchmark framework:

1. Place `ITSMixer.py` in `models/`
2. Replace `exp/exp_main.py` with the provided `exp_main_itsmixer.py`

The modified `exp_main.py` adds safe import + registration of ITSMixer with minimal changes to the original framework.

---

## Acknowledgments

This implementation builds on the excellent [tsf-new-paper-taste](https://github.com/hughxx/tsf-new-paper-taste) benchmark framework for training infrastructure, data loading, and evaluation. We only integrated our model — no core training code was altered.

We also utilize the RevIN normalization technique from the time series literature.

---

## Citation

If you use ITSMixer in your research, please cite:

```bibtex
@article{lotfi2026itsmixer,
  title = {ITSMixer: iterative time-mixing MLPs for efficient long-term forecasting},
  author = {Lotfi, Arian and Damari, Siavash},
  journal = {Evolving Systems},
  year = {2026},
  doi = {10.1007/s12530-026-09830-0},
  url = {https://link.springer.com/article/10.1007/s12530-026-09830-0}
}
```

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

**Note**: The training infrastructure is inherited from the tsf-new-paper-taste framework (check their repository for licensing).

---

- Open an issue: [https://github.com/JZX100II/ITSMixer/issues](https://github.com/JZX100II/ITSMixer/issues)
- Read the paper: [Springer Link](https://link.springer.com/article/10.1007/s12530-026-09830-0)
```
