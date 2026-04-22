# MATH 562 Final Project: Theory of Machine Learning

## Project Description

This project tests theoretical results from MATH 562 (Theory of Machine Learning), with an emphasis on overparameterized models. It contains two components:

1. **Linear Regression and Joint Scaling Limits**: Explore the concentration of loss curves in joint scaling limits for ridge linear regression, the relationship between generalization and training dynamics, and different SGD variants (streaming, multi-pass, single/multiple shuffling, momentum, batch size effects).
2. **Scaling Regimes in Two-Layer Neural Networks**: Implement and analyze three fundamental training regimes for two-layer neural networks — Neural Tangent Kernel (NTK), Mean Field (MF), and Random Features (RF) — comparing their training dynamics, learning rate scalings, and kernel evolution.

For each component, we formulate hypotheses based on theory, design experiments to test them, and interpret the results in light of the predictions from class.

## Environment

- Python 3.13+
- Required libraries (see `requirements.txt`)
- Jupyter support required for running `.ipynb` notebooks

## User Manual

Create a Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Install Jupyter support if needed:

```bash
pip install notebook ipykernel
```
then in VS Code select the Python kernel associated with your virtual environment to run `.ipynb` files.



## Component 1: Linear Regression and Joint Scaling Limits

The component 1 folder is structured as follows:
- `main.py`: Core implementation of SGD algorithms, risk computations, and theoretical scaling quantities.
  It includes:
  - SGD variants: `run_sgd`, `single_shuffle_sgd`, `multiple_shuffle_sgd`, `sgd_momentum_fixed_delta`, `sgd_batch`
  - Risk functions: `compute_empirical_risk`, `compute_true_risk`
  - Scaling terms: `compute_first_term`, `compute_second_term`, `compute_third_term_sgd`
  - Visualization and experiments: `plot_risk`, `estimate_threshold`
- `experiment_component1.ipynb`: Driver notebook that sets parameters and calls functions from `main.py` to run all experiments.

### Experiment 1: Scaling laws in small-batch SGD

This experiment studies empirical scaling laws in ridge regression under SGD by evaluating how gradient, Hessian, and stochastic noise terms scale with dimension.

It uses:
- `main.compute_first_term`
- `main.compute_second_term`
- `main.compute_third_term_sgd`

and visualizes results using:
- `main.plot_risk`

---

### Experiment 2: Effect of randomness in SGD

Executed in the same notebook.

We compare convergence under different sampling schemes using:
- `main.single_shuffle_sgd`
- `main.multiple_shuffle_sgd`

with evaluation via:
- `main.plot_risk`

---

### Experiment 3: SGD with momentum

Executed in the same notebook.

We analyze momentum dynamics using:
- `main.sgd_momentum_fixed_delta`

with results plotted using:
- `main.plot_risk`

---

### Experiment 4: Small vs large batch sizes

Executed in the same notebook.

We study batch-size scaling effects using:
- `main.sgd_batch`

with different batch schedules `batch_size = n^η`, and visualization via:
- `main.plot_risk`

---

### Extra Experiment: Threshold estimation

Executed in the same notebook.

We estimate transition thresholds between stable and unstable regimes using `main.estimate_threshold`.

---

All experiments are controlled from `experiment_component1.ipynb`, which acts as a parameter interface over functions defined in `main.py`.



## Component 2: Scaling Regimes in Two-Layer Neural Networks
The component 2 folder is structured as follows:
- `data_generator.py`: Utility for generating datasets (synthetic, MNIST, CIFAR-10)
- `regimes.py`: Implementations of the three training regimes (MF, NTK, RF) as well as the two-layer neural network architecture.
- `utils.py`: Utility functions for training, evaluation, logging, and plotting.
- `checkpoints.py`: Functions for saving and loading run checkpoints.
- `exp_1/`: Performance comparison across regimes (MF, NTK, RF)
- `exp_2/`: Learning rate scaling analysis
- `exp_3/`: Kernel evolution and constancy

### Experiment 1: Performance comparison across regimes (MF, NTK, RF)
```
python component2/exp_1/main.py
```
Specific configurations can be made through arguments (run --help for details and defaults). 
The results can be found in the results folder, which contains the logs, plots, and JSON output files.

### Experiment 2: Learning rate scaling analysis

```
python component2/exp_2/main.py
```
The results can be found in the results folder, which contains the logs, plots, and JSON output files.

### Experiment 3: Kernel evolution and constancy

```
python component2/exp_3/main.py
```
By default, this runs the experiment over:
regimes: NTK, MF, RF,
widths: m ∈ {100, 200, 400, 800},
activations: tanh (primary),
checkpoints during training for kernel evaluation,
multiple learning rate scalings (beta_scalings).


Optional arguments (see main.py) allow you to modify:
--m_values,
--activation,
--n_iters,
--log_every,
--n_kernel,
--beta_scalings.
The results can be found in the results/ folder, which contains logs, plots, JSON output files, and saved kernel matrices.
