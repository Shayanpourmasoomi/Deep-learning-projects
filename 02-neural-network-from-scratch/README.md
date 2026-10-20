# A Neural-Network Framework from Scratch in NumPy

A modular, fully connected neural-network library written in **NumPy only**: forward and backward passes for each layer, numerical gradient checking, and several optimisers. It is then used for a real regression task, predicting median house values on the California Housing dataset.

## What is implemented

| Component | Details |
|---|---|
| **Layers** | Affine (fully connected), ReLU, sigmoid. Each has a forward pass and an analytic backward pass |
| **Loss** | Mean squared error and its gradient |
| **Network** | Multi-layer fully connected network with arbitrary depth and width |
| **Verification** | Numerical gradient checks against every analytic gradient |
| **Optimisers** (`utils/optim.py`) | SGD, SGD with momentum, RMSProp, Adam |
| **Training loop** (`utils/solver.py`) | Mini-batch training, learning-rate decay, train/validation tracking |

## Application

A multi-layer network trained on California Housing (20,640 districts, 8 features), with training and validation error tracked across epochs. Diagnostic plots compare predictions with targets.

## Files

```
neural_network_from_scratch.ipynb   # layer implementations, gradient checks, training, evaluation
utils/                              # optimisers, solver, data loading, gradient-check helpers
california_housing.csv              # dataset (StatLib)
```

## Run

```bash
pip install numpy matplotlib pandas
jupyter notebook neural_network_from_scratch.ipynb
```
