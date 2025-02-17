# Masked Self-Attention Experiment in PyTorch

This repository contains a Jupyter Notebook and a tokenizer configuration file (`tokenizer.json`) for training a small masked self-attention model for text inference using PyTorch. This project served as an initial deep learning experiment before progressing to more advanced transformer-based models and large language models (LLMs).

---

## Contents

- **masked_self_attention.ipynb**: The main notebook containing code for data processing, model definition (including masked self-attention), training (with mixed precision using AMP and GradScaler), and evaluation.
- **tokenizer.json**: A JSON file with your tokenizer configuration or saved tokenizer data.
- **README.md**: This file.

---

## Project Overview

In this experiment, I implemented a small Transformer-like model with masked self-attention to perform text inference. The model includes:
- An **Embedding** layer to convert word indices into vector representations.
- A **Masked Self-Attention** module that computes self-attention while preventing tokens from attending to future positions.
- A **Feed-Forward Network (FFN)** with two linear layers and a ReLU activation in between.
- A final **Linear Layer** that produces output logits corresponding to each token in the vocabulary.

The notebook also demonstrates:
- The use of PyTorch's Automatic Mixed Precision (AMP) with a GradScaler.
- Basic logging to measure data loading and GPU compute times during training.
- How to create and apply a mask to the attention scores to enforce causality (i.e., preventing the model from “seeing” future tokens).
