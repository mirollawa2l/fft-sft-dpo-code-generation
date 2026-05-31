# CodeGen Alignment Lab

An end-to-end exploration of Large Language Model adaptation and alignment techniques for Python code generation.

This project compares three increasingly advanced training approaches:

1. **Full Fine-Tuning (FFT)** using XLM-RoBERTa adapted for causal language modeling.
2. **Supervised Fine-Tuning (SFT)** using QLoRA on Qwen2-1.5B-Instruct.
3. **Direct Preference Optimization (DPO)** for post-training alignment and safety improvement.

The goal is to investigate how different training strategies affect code generation quality, instruction following, and safety behavior.

---

## Project Overview

Large Language Models can be adapted to specialized tasks through multiple stages of training.

This project implements and compares:

### Part I — Full Fine-Tuning (FFT)

- Uses XLM-RoBERTa Base (278M parameters).
- Converts an encoder-only multilingual model into a Causal Language Model.
- Fine-tunes all model parameters end-to-end.
- Trained on Python instruction-response pairs.
- Evaluated using loss and perplexity.

### Part II — Supervised Fine-Tuning (SFT) with QLoRA

- Uses Qwen2-1.5B-Instruct.
- Applies 4-bit quantization using BitsAndBytes.
- Uses LoRA adapters for parameter-efficient fine-tuning.
- Trains only a small subset of parameters.
- Reduces GPU memory requirements while maintaining strong performance.

### Part III — Direct Preference Optimization (DPO)

- Starts from the SFT model.
- Uses preference pairs from the Truthy-DPO dataset.
- Aligns model outputs toward preferred responses.
- Improves safety and instruction-following behavior.
- Experiments with multiple β values to study alignment trade-offs.

---

## Dataset

### Python Instruction Dataset

Dataset:
`flytech/python-codes-25k`

Each sample contains:

- Instruction
- Optional Input
- Expected Output

The dataset is converted into instruction-following training examples.

Example:

```text
Instruction:
Write a Python function to calculate factorial

Response:
def factorial(n):
    ...
```

### Preference Dataset

Dataset:
`jondurbin/truthy-dpo-v0.1`

Used for DPO alignment with:

- Prompt
- Chosen response
- Rejected response

---

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- PEFT
- LoRA
- QLoRA
- TRL
- BitsAndBytes
- Accelerate
- Weights & Biases

---

## Training Pipeline

```text
Python Dataset
      │
      ▼
 Full Fine-Tuning
 (XLM-RoBERTa)
      │
      ▼
 Supervised Fine-Tuning
 (Qwen2-1.5B + QLoRA)
      │
      ▼
 Direct Preference Optimization
      │
      ▼
 Aligned Code Assistant
```

---

## Experimental Results

### Full Fine-Tuning

- End-to-end training of all parameters.
- Evaluated using:
  - Validation Loss
  - Perplexity

### QLoRA Fine-Tuning

- 4-bit quantized model.
- LoRA adapters trained.
- Significant reduction in memory usage.
- Suitable for resource-constrained environments.

### DPO Alignment

Tested multiple β values:

| β   | Observation              |
| --- | ------------------------ |
| 0.1 | Unstable learning        |
| 0.5 | Best overall performance |
| 0.8 | Overconfident behavior   |
| 1.0 | Too conservative         |

Best performance achieved with:

```text
β = 0.5
```

---

## Safety Evaluation

The project compares model behavior on:

### Safe Requests

- Prime number checker
- Factorial implementation
- Algorithm explanations

### Unsafe Requests

- Keylogger generation
- Credential harvesting
- Malicious automation

DPO alignment is used to encourage safer responses while preserving coding capabilities.

---

## Key Learnings

- Full Fine-Tuning provides maximum flexibility but requires significant compute resources.
- QLoRA dramatically reduces memory requirements while retaining strong performance.
- DPO effectively aligns model behavior without expensive reinforcement learning.
- The β parameter plays a critical role in balancing capability and alignment.
