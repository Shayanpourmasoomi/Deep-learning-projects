# Parameter-Efficient Fine-Tuning of an LLM with QLoRA

Fine-tuning **Qwen2.5-0.5B-Instruct** with **QLoRA**: 4-bit quantised base weights plus trainable low-rank adapters. The project compares adapter ranks, specialised adapters for individual tasks, and a single general adapter trained on a task mixture under a matched compute budget.

## Setup

| | |
|---|---|
| **Base model** | Qwen2.5-0.5B-Instruct, frozen and loaded in 4-bit (bitsandbytes) |
| **Adapters** | LoRA via `peft`; at rank 16, 8.8 M trainable parameters (1.75% of the model) |
| **Tasks** | GSM8K (grade-school maths), CommonsenseQA (multiple choice), AG News (topic classification) |
| **Metric** | Validation exact match |

## Results

**LoRA rank ablation (CommonsenseQA):**

| Rank | 4 | **16** | 64 |
|---|---|---|---|
| Exact match | 0.58 | **0.61** | 0.57 |

Rank 16 was chosen. Higher rank does not help at this data scale, and rank 64 starts to overfit.

**Specialised adapters (rank 16, 300 examples each):**

| Task | Exact match |
|---|---|
| AG News | 0.86 |
| CommonsenseQA | 0.61 |
| GSM8K | 0.25 |

Multi-step arithmetic reasoning (GSM8K) is by far the hardest task for a 0.5B-parameter model, even after adaptation. The notebook also covers the theory of full fine-tuning vs PEFT, and trains a general adapter on the mixture of all three tasks with the same number of optimiser updates.

## Files

```
qlora_lora_finetuning.ipynb   # theory, data formatting, QLoRA training, ablation, evaluation
```

## Run

```bash
pip install torch transformers peft bitsandbytes datasets accelerate
```

Requires a CUDA GPU.
