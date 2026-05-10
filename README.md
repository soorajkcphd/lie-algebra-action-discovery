# Discovering Lie-Algebraic Action Spaces for Sample-Efficient Reinforcement Learning

Code for the ICECCME 2026 paper:

> Sooraj K.C and Vivek Mishra.
> "Discovering Lie-Algebraic Action Spaces for Sample-Efficient Reinforcement Learning in Latent Spaces."
> Proc. ICECCME 2026, Bali, Indonesia, Oct 15-17 2026.

---

## Summary

Standard RL policies parameterize arbitrary linear transformations, requiring d^2 action parameters.
This paper shows that the Lie algebra governing task-relevant transformations can be discovered from
data using matrix logarithms and SVD, reducing the action space to k parameters (k << d^2).

The discovered algebra directly constrains the RL policy:
- 95% parameter reduction (k=3 vs d^2=64 for d=8)
- Matches oracle ground-truth performance (p=1.000, Welch's t-test)
- Significantly outperforms PCA (p=0.039) and random baselines (p=0.013)

---

## Repository Structure

```
lie-algebra-action-discovery/
    run_full_experiments.py     # Main experiment script (all 5 experiments)
    statistical_tests.py        # Welch's t-tests on per-seed results
    requirements.txt            # Python dependencies
    README.md                   # This file
    results/                    # Generated after running experiments
        exp2_per_seed_results.json
        exp2_per_seed_results.csv
        statistical_test_results.json
        fig1_concept.pdf
        fig2_pipeline.pdf
        fig3_structure_constants.pdf
        fig4_learning_curves.pdf
        fig5_performance_profiles.pdf
        fig_experiments.pdf
```

---

## Setup

Python 3.8+ required.

```bash
git clone https://github.com/soorajkcphd/lie-algebra-action-discovery
cd lie-algebra-action-discovery
pip install -r requirements.txt
```

---

## Running the Experiments

### Step 1: Run all experiments

```bash
python run_full_experiments.py
```

This runs all 5 experiments across 5 seeds and saves:
- Per-seed results to results/exp2_per_seed_results.json and .csv
- All 6 figures to results/
- LaTeX tables printed to stdout

Estimated runtime: 15-30 minutes on a modern GPU (NVIDIA GeForce RTX 5060 Ti used in the paper).

### Step 2: Run statistical tests

```bash
python statistical_tests.py
```

Reads results/exp2_per_seed_results.json automatically and runs Welch's t-tests.
Outputs t-statistics, p-values, and Cohen's d for all pairwise comparisons.
Saves results to results/statistical_test_results.json.

---

## Experiments

| Exp | What it tests |
|-----|---------------|
| 1   | Structure discovery: does the algorithm recover the correct Lie algebra? |
| 2   | RL comparison: 5-way comparison (Discovered vs PCA vs Full vs Random vs GT) |
| 3   | Noise sensitivity: how does discovery degrade with observation noise? |
| 4   | Dimension mismatch: what if the wrong k is used? |
| 5   | Subspace recovery: does discovery help even without Lie structure? |

All experiments use the ground-truth algebra so(2) (+) R^2 embedded in R^{8x8}.

---

## Key Results (5 seeds, mean +/- std)

| Method       | Params | Success (%) |
|--------------|--------|-------------|
| Full (d^2)   | 64     | 36.0 +/- 5.7 |
| Random (k)   | 3      | 32.0 +/- 5.4 |
| PCA (k)      | 3      | 36.0 +/- 3.3 |
| Discovered   | 3      | 47.3 +/- 7.7 |
| Ground Truth | 3      | 47.3 +/- 5.3 |

Statistical tests (Welch's t-test, n=5 seeds):
- Discovered vs Ground Truth: p=1.000 (not significant -- methods are equivalent)
- Discovered vs PCA:          p=0.039, Cohen's d=1.71 (significant)
- Discovered vs Random:       p=0.013, Cohen's d=2.06 (significant)

---

## Hardware

Experiments were run on:
- NVIDIA GeForce RTX 5060 Ti (8 GB VRAM, CUDA 13.2)
- Ubuntu 24, Python 3.10, PyTorch 2.x

---

## Citation

If you use this code, please cite:

```
@inproceedings{kc2026lie,
  title     = {Discovering Lie-Algebraic Action Spaces for Sample-Efficient
               Reinforcement Learning in Latent Spaces},
  author    = {Sooraj K.C and Vivek Mishra},
  booktitle = {Proc. Int. Conf. Electrical, Computer, Communications and
               Mechatronics Engineering (ICECCME)},
  year      = {2026},
  address   = {Bali, Indonesia}
}
```

---

## Contact

Sooraj K.C -- ksoorajPHD23@sam.alliance.edu.in
Dept. of Pure and Applied Mathematics, Alliance University, Bengaluru, India
