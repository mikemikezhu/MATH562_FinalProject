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

### Experiment 1: Performance comparison across regimes (MF, NTK, RF)

TODO

### Experiment 2: Learning rate scaling analysis

```
python component2/exp_2/main.py
```
The results can be found in the results folder, which contains the logs, plots, and JSON output files.

### Experiment 3: Kernel evolution and constancy

TODO
