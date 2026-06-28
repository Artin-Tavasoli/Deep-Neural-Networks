
# Disaster Tweet Classification with a Tiny Transformer Encoder

  

This repository contains an implementation of a custom Tiny Transformer Encoder built entirely from scratch using PyTorch. It explores self-attention mechanisms, positional encodings, and modular transformer architectures to classify tweets as either real disasters or non-disasters.

  

---

  

## 1. Tiny Transformer Encoder from Scratch

  

* Implemented a custom Transformer Encoder architecture to classify disaster-related tweets without relying on pre-built Transformer libraries.

  

### ⚙️ Methodology

  

* **Text Preprocessing:** Cleaned the raw tweet data by normalizing text, removing URLs, emojis, HTML tags, and stopwords, and replacing common abbreviations.

* **Architecture:** Constructed a modular PyTorch model featuring:

* **Token & Positional Embeddings:** Mapped input sequences to dense vectors and injected sinusoidal positional encodings to retain sequence order.

* **Multi-Head Self-Attention:** Implemented Scaled Dot-Product Attention from scratch, reshaping tensors to compute multiple attention heads in parallel.

* **Encoder Blocks:** Built feed-forward networks, layer normalization, and dropout layers around the attention outputs.

* **Classifier Head:** Fed the output of a specialized `[CLS]` token into a dense neural network for binary classification.

  
  

* **Class Imbalance Handling:** Addressed the dataset's target imbalance by incorporating class weights into the `CrossEntropyLoss` function, ensuring the model focuses on semantic understanding rather than frequency bias.

  

### 📊 Results

  

The custom model successfully learned to distinguish between literal and figurative uses of disaster-related keywords.

  

* **Self-Attention Visualization:** Extracted and plotted the attention weights, proving that the model successfully learned to assign higher attention scores to critical tokens like "emergency", "damage", and "storm", while ignoring irrelevant filler words.

  

---

  

## 2. BERT Fine-Tuning & Comparison

  

* Evaluated the custom model against a pre-trained BERT language model to understand the impact of large-scale contextual pre-training.

  

### ⚙️ Methodology

  

* **Transfer Learning:** Utilized a pre-trained BERT model and fine-tuned it on the disaster tweets dataset to establish a performance baseline.

* **Evaluation:** Compared the custom Tiny Transformer against the BERT model using confusion matrices, accuracy, and loss curves.

  

### 📊 Results

  
While the Tiny Transformer proved the viability of the from-scratch architecture, the pre-trained BERT model leveraged its massive prior knowledge of contextual language to achieve superior structural understanding and classification accuracy, minimizing false positives on figurative language.
