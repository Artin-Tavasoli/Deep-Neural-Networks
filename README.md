# Deep Neural Networks

## 📌 Overview

This repository contains a comprehensive collection of implementations and research reports for the **Deep Neural Networks** course. The projects explore the theoretical foundations and practical applications of modern deep learning architectures, ranging from foundational Multi-Layer Perceptrons (MLPs) and Autoencoders to advanced architectures like Sparse R-CNNs and Transformers.

Each project includes a dedicated report, mathematical derivations, and a self-contained PyTorch implementation.

---

## 📂 Projects

| Project | Description | Key Topics |
| --- | --- | --- |
| **1. Autoencoders & MLP Regression** | Implementation of foundational architectures, including dimensionality reduction on the MNIST dataset via **Autoencoders** and complex life expectancy prediction using deep **MLPs**. | `Autoencoders` `MLP` `Regression` `Dimensionality Reduction` |
| **2. Face Recognition & Angular Margin** | Comparison of a standard Softmax CNN against an architecture optimized with **Angular Margin Loss**. Explores feature space geometry and inter-class variance on a custom synthetic dataset. | `CNN` `Angular Margin Loss` `Feature Geometry` `Face Recognition` |
| **3. RNNs & Sentiment Analysis** | Implementation of **Vanilla RNN, LSTM, and GRU** models for movie review sentiment analysis. Features mathematical and empirical gradient diagnostics to analyze **BPTT** and the vanishing gradient problem. | `RNN` `LSTM` `GRU` `BPTT` `Sentiment Analysis` |
| **4. Sparse R-CNN Object Detection** | A lightweight **Sparse R-CNN** built from scratch for localizing geometric shapes. Utilizes learnable proposal boxes, dynamic heads, and **Bipartite Matching** (Hungarian algorithm) instead of NMS. | `Sparse R-CNN` `Object Detection` `Bipartite Matching` `GIoU Loss` |
| **5. Transformers & BERT Classification** | Construction of a custom **Tiny Transformer Encoder** from scratch for classifying disaster tweets, featuring custom self-attention. Includes a comparative evaluation against a fine-tuned pre-trained **BERT** baseline. | `Transformers` `Self-Attention` `BERT` `NLP` |

*Note: Please refer to the individual project folders for detailed READMEs.*