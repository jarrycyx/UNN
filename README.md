# UNN - Official Repository for Causal Neural Network

## Overview
This repository provides our latest research on Causal Neural Network. 

| Algorithm | Summary         | Paper |
|--------|---------------------------------------------------------------------------|----|
| CUTS  | EM-Style joint causal graph learning and missing data imputation for irregular temporal data |  [ICLR 2023](https://openreview.net/forum?id=UG8bQcD3Emv)  |


## 🚩 CUTS: Neural Causal Discovery from Irregular Time-Series Data

[Paper](https://openreview.net/forum?id=UG8bQcD3Emv) | [Tutorial ![Open filled In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/) 

This is the official repository of the paper **CUTS: Neural Causal Discovery from Irregular Time-Series Data**.

### ✍️ Paper summary

<center><img src="CUTS/figures/method_mix.jpg" width="800px"></center>

Causal discovery from time-series data has been a central task in machine learning. Recently, Granger causality inference is gaining momentum due to its good explainability and high compatibility with emerging deep neural networks. However, most existing methods assume structured input data and degenerate greatly when encountering data with randomly missing entries or non-uniform sampling frequencies, which hampers their applications in real scenarios.  

To address this issue, here we present **CUTS**, a neural Granger causal discovery algorithm to jointly impute unobserved data points and build causal graphs, via plugging in two mutually boosting modules in an iterative framework:
1. **Latent data prediction stage:** designs a Delayed Supervision Graph Neural Network (DSGNN) to hallucinate and register irregular data which might be of high dimension and with complex distribution;
2. **Causal graph fitting stage:** builds a causal adjacency matrix with imputed data under sparse penalty.

Experiments show that CUTS effectively infers causal graphs from irregular time-series data, with significantly superior performance to existing methods. *Our approach constitutes a promising step towards applying causal discovery to real applications with non-ideal observations.*

### Requirements

Please see [requirements.txt](requirements.txt).

### Example

<center><img src="CUTS/figures/example.jpg" width="800px"></center>

We created a simple tutorial that addresses causal discovery on a three-node dataset [`cuts_example.ipynb`](CUTS/cuts_example.ipynb). 

More examples will be available in the future.


### 😘 Citation
If you use this code, please consider citing [our work](https://openreview.net/forum?id=UG8bQcD3Emv).
