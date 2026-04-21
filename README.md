# MATH 562 Final Project: Theory of Machine Learning

## Project Description

This project tests theoretical results from MATH 562 (Theory of Machine Learning), with an emphasis on overparameterized models. It contains two components:

1. **Linear Regression and Joint Scaling Limits**: Explore the concentration of loss curves in joint scaling limits for ridge linear regression, the relationship between generalization and training dynamics, and different SGD variants (streaming, multi-pass, single/multiple shuffling, momentum, batch size effects).
2. **Scaling Regimes in Two-Layer Neural Networks**: Implement and analyze three fundamental training regimes for two-layer neural networks — Neural Tangent Kernel (NTK), Mean Field (MF), and Random Features (RF) — comparing their training dynamics, learning rate scalings, and kernel evolution.

For each component, we formulate hypotheses based on theory, design experiments to test them, and interpret the results in light of the predictions from class.

## Environment

- Python 3.13+
- Required libraries (see `requirements.txt`)

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

## Component 1: Linear Regression and Joint Scaling Limits

### Experiment 1: Scaling of the learning rate in small-batch SGD

TODO

### Experiment 2: Effect of randomness in SGD (uniform sampling vs. single shuffle vs. multiple shuffle)

TODO

### Experiment 3: SGD with momentum

TODO

### Experiment 4: Small vs. large batch sizes

TODO

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
