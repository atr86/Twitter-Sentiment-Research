# Efficient & Robust Sentiment Analysis: GloVe-CNN-BiLSTM-MHA-SVM

This repository contains a hybrid neural network architecture designed for high-efficiency sentiment classification. By combining CNNs for feature extraction and LSTMs for sequence modeling, this pipeline achieves strong performance with significantly lower computational overhead than Transformer models.

## 🧠 Model Architecture

The pipeline is designed to bypass the $O(n^2)$ computational bottleneck of deep self-attention while maintaining high accuracy:
*   **GloVe Embeddings:** Static $O(1)$ lookups for rich semantic initialization, avoiding dynamic contextual embedding overhead.
*   **1D Convolutional Neural Network (CNN):** Acts as a local feature extractor, rapidly reducing the spatial dimensionality of noisy text (e.g., typos, n-grams).
*   **Bidirectional LSTM (BiLSTM):** Captures sequential context and long-term dependencies from both directions.
*   **Multi-Head Attention (MHA):** A single targeted attention layer applied to the recurrent outputs, weighting the most critical sentiment-bearing tokens.
*   **Support Vector Machine (SVM):** Replaces the standard Softmax layer to provide a robust, maximum-margin decision boundary, preventing overfitting on smaller datasets.

## 🚀 Training Environment

The models in this repository were developed and trained using the following specifications:

* **Platform**: Kaggle.
* **Hardware**: NVIDIA T4 GPU.
* **Optimization**: Early stopping implemented, to prevent overfitting based on validation loss and F1 score trends.

## 📊 Datasets

You can access the datasets used for this project at the following links:

* [**Sentiment140**](https://www.kaggle.com/datasets/kazanova/sentiment140)
* [**Tweets Sentiment Analysis**](https://huggingface.co/datasets/bdstar/Tweets-Sentiment-Analysis)
* [**Glove Embedding**](https://www.kaggle.com/datasets/wiseyy/glove6b300dtxt)

## 📁 Repository Structure

The repository is organized into four primary notebooks, split between the large-scale baseline and the cross-domain robustness test.

### 1. Sentiment140 Pipeline (Baseline & Scale)

* `sentiment140-glove-cnn-bilstm-attn-mlp.ipynb`: Handles data loading, preprocessing, and training the core deep learning architecture with an MLP head.
* `sentiment140-svm.ipynb`: Implements the Support Vector Machine (SVM) classification head to optimize the decision boundary.

### 2. HuggingFace Pipeline (Domain Shift Test)

* `hf-glove-cnn-bilstm-attn-mlp.ipynb`: Trains the model on organic human dialogue sentences to establish a high-accuracy baseline.
* `hf-svm.ipynb`: Evaluates the model against synthetic, AI-generated text to measure robustness against distribution shifts.

## 📉 Key Results

* **Validation Accuracy**: Achieved **87.4481%** on human-generated text.
* **Cross-Domain Performance**: Maintained **67.39%** accuracy when tested on synthetic ChatGPT-generated text, demonstrating successful zero-shot generalization despite a significant domain shift.
* **Computational Efficiency**: Training from scratch on 1.6M records was completed in time required for training of 1 epoch of standard Transformer (e.g., BERT).

## 🛠️ Usage

1. Clone the repository.
2. Download pre-trained GloVe embeddings.
3. Execute the training notebooks to generate feature weights, followed by the SVM notebooks for final classification.

**Setup:**
1. Clone the repository:
   ```bash
   git clone https://github.com/atr86/Twitter-Sentiment-Research.git
   cd Twitter-Sentiment-Research
   ```
2. Load dataset/model
3. Run Notebook

**Prerequisites:**
*   Python 3.8+
*   Pytorch
*   Scikit-learn, Pandas, NumPy, Matplotlib
---


### Why use a hybrid CNN-BiLSTM-MHA-SVM?

This architecture is designed to provide a "middle ground" in NLP: more powerful than simple RNNs but significantly faster than Transformers. It is particularly effective for researchers or developers working with hardware constraints who still require state-of-the-art attention mechanisms.

---

TSA-Net: A Hybrid Convolutional Neural Network and Bidirectional LSTM Model with Multi-Head Attention for Large-Scale Twitter Sentiment Analysis
MD Soyeb Hoque, Atrij Roy, Anumit Kumar Jana and Pawan Kumar Singh,* [0000-0002-9598-7981] (2026), 
*State-of-the-Art Techniques in Text Mining for Healthcare Data Analysis and Prediction*, Springer Nature, Springer (Scopus Indexed). Submitted June, 2026.

