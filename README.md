# 🧠 LLM from Scratch: Building a Mini-GPT from First Principles

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yaniv-11/mini-GPT/blob/main/llm-from-scratch.ipynb)

**A comprehensive implementation of a modern Large Language Model built entirely from scratch using PyTorch.** This educational project breaks down the complex architecture of GPT-style transformers into understandable, modular components with detailed explanations and visualizations.

## 📋 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Implementation Details](#-implementation-details)
- [Training Pipeline](#-training-pipeline)
- [Results](#-results)
- [Contributing](#-contributing)
- [License](#-license)
- [References](#-references)
- [Acknowledgements](#-acknowledgements)

## 🎯 Overview

This project implements a complete **decoder-only transformer architecture** (GPT-style) from the ground up, including tokenization, embeddings, multi-head attention, feed-forward networks, and the complete training/inference pipeline. The implementation is designed to be **educational**—every component is explained with mathematical foundations, code implementations, and visualizations.

Unlike using high-level libraries like Hugging Face Transformers, this project builds every layer manually to provide deep understanding of how modern LLMs work internally.

## ✨ Key Features

### 🔧 Core Components
- **Byte Pair Encoding (BPE) Tokenizer** - Custom implementation with merge rules and vocabulary management
- **Learned Positional Embeddings** - Both absolute and relative positional encoding strategies
- **Multi-Head Causal Self-Attention** - With efficient key-value caching for generation
- **Transformer Block** - Complete with layer normalization and residual connections
- **GELU Activation** - Gaussian Error Linear Units as used in GPT models
- **RMSNorm** - Root Mean Square Layer Normalization (modern variant)

### 📚 Educational Value
- **Step-by-step explanations** of each mathematical operation
- **Visualizations** of attention patterns, gradient flow, and embedding spaces
- **Interactive experiments** to modify hyperparameters and observe effects
- **Performance comparisons** between different implementations
- **Debugging utilities** for understanding model internals

### 🚀 Production-Ready Features
- **Mixed Precision Training** (FP16/FP32)
- **Gradient Accumulation** for large batch training
- **Model Checkpointing** and automatic resumption
- **TensorBoard Integration** for training visualization
- **Export to ONNX** for deployment

## 🏗️ Architecture

```ascii
Input Text
    │
    ▼
Tokenizer (BPE)
    │
    ▼
Token Embeddings + Positional Encoding
    │
    ▼
┌─────────────────────────────────────┐
│  Transformer Decoder Stack (N layers)│
│  ┌─────────────────────────────┐    │
│  │ Multi-Head Causal Attention │    │
│  │   • Query/Key/Value Proj    │    │
│  │   • Scaled Dot-Product      │    │
│  │   • Causal Masking          │    │
│  └─────────────────────────────┘    │
│  │       LayerNorm 1           │    │
│  └─────────────────────────────┘    │
│  │  Feed-Forward Network       │    │
│  │   • GELU Activation         │    │
│  │   • Linear Projections      │    │
│  └─────────────────────────────┘    │
│  │       LayerNorm 2           │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
    │
    ▼
Final Layer Normalization
    │
    ▼
Vocabulary Projection (Logits)
    │
    ▼
Softmax → Next Token Prediction
