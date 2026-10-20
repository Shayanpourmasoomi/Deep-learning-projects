# Self-Supervised Representation Learning with SimCLR

An implementation of **SimCLR** (Chen et al., 2020), pre-trained on the unlabelled split of **STL-10**: a ResNet encoder learns image representations without labels by pulling together two augmented views of the same image and pushing apart views of different images. Representation quality is then measured with a linear classifier trained on top of the frozen encoder.

## Components

| Component | Details |
|---|---|
| **Augmentations** | Random resized crop, flip, colour jitter with strength *s*, random greyscale, Gaussian blur |
| **View generator** | Two independent augmentations per image form each positive pair |
| **Encoder** | ResNet backbone *f(·)* with an MLP projection head *g(·)* |
| **Loss** | NT-Xent (normalised temperature-scaled cross-entropy), implemented from scratch |
| **Evaluation** | Linear probe on frozen features |

## Results

- Contrastive pre-training: instance-discrimination top-1 accuracy of about 65% within the batch after 9 epochs
- Linear probe on frozen features: **62.8% top-1 / 97.3% top-5** test accuracy after 20 epochs, using no labels during pre-training

## Files

```
simclr_contrastive_learning.ipynb   # augmentations, model, NT-Xent loss, pre-training, linear evaluation
```

## Run

```bash
pip install torch torchvision tqdm
```

A GPU is strongly recommended for pre-training.

## Reference

Chen, T., Kornblith, S., Norouzi, M., and Hinton, G. (2020). A Simple Framework for Contrastive Learning of Visual Representations. *ICML*.
