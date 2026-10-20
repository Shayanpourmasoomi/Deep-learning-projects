# PyTorch Fundamentals

A ground-up tour of PyTorch: tensors, broadcasting, GPU execution, data pipelines and training loops. It ends with two small networks: a regressor fitted to a 1-D function and an MLP classifier on MNIST.

## Contents

| Part | Topics |
|---|---|
| **Tensors** | Construction, dtypes, indexing, reshaping, element-wise and reduction operations, broadcasting |
| **Hardware** | Moving computation to the GPU and timing CPU against GPU |
| **Data** | `Dataset` / `DataLoader`, CIFAR-10 loading and batching, saving and restoring models |
| **Function approximation** | A small network trained to fit a noisy 1-D function, with loss curves over 1,000 epochs |
| **MNIST classification** | MLP classifier (training loss 0.43 → 0.09 in 5 epochs) and t-SNE of hidden-layer activations, showing class clusters forming in the output layer |

## Files

```
pytorch_fundamentals.ipynb   # all exercises, outputs and plots
pytorch_basic.py             # helper functions used by the notebook
```

## Run

```bash
pip install torch torchvision matplotlib scikit-learn
jupyter notebook pytorch_fundamentals.ipynb
```
