# Text Classification with Traditional and Transformer-Based Embeddings

## Project Overview
This project benchmarks multiple natural language processing (NLP) approaches for **text classification** on a dataset of ~34 k book blurbs.
The objective was to compare traditional embedding-based and sequence models against modern transformer-based architectures.

Each model was trained on the same preprocessed dataset and evaluated on test accuracy.

---

## Model Performance Summary

| Model Type | Configuration Highlights | Test Accuracy |
|-------------|--------------------------|----------------|
| **GloVe 50D** | 50-dimensional GloVe (6B) | **71.19 %** |
| **GloVe 100D** | 100-dimensional GloVe (6B) | **71.51 %** |
| **GloVe 200D** | 200-dimensional GloVe (6B) | **74.73 %** |
| **GloVe 300D** | 300-dimensional GloVe (6B) | **76.15 %** |
| **GloVe 300D (tuned)** | Hyperparameter-tuned: epochs=50, batch=64, lr=0.0005, hidden_dim=128, patience=7 | **77.30 %** |
| **Sentence‑Transformer** | 384-dimensional sentence vectors (`all‑MiniLM‑L6‑v2`) | **76.30 %** |
| **Sentence‑Transformer (tuned)** | Hyperparameter-tuned: epochs=50, batch=64, lr=0.002, hidden_dim=256, warmup_ratio=0.1, patience=3 | **77.66 %** |
| **GRU (various models tuned)** | 300D GloVe inputs. Tuning includes: Bi‑GRU + dropout + attention + weight decay + hidden dim 128 -> 64 | **74 – 76.76%** |
| **DistilBERT** | Transformer fine‑tuned on task | **82.22 %** |


---

## Experimental Details

### GloVe Models
- Used pre-trained **GloVe 6B** embeddings of 50–300 dimensions.  
- Classifier: feed-forward network (hidden_dim = 128).  
- **Training:** 50 epochs, batch size 32, learning_rate 0.001, patience 3.  
- Performance improved steadily with embedding size (plateaued around 300 D).


## Grid Search Hyperparameter Tuning (GloVe Models)

- Ran **two grid searches** using the parameter grid below, with **patience values of 3 and 7** to evaluate their effect on model performance.

**Parameter Grid**
```python
param_grid = {
    "lr": [0.001, 0.0005, 0.002],
    "batch_size": [32, 64],
    "hidden_dim": [128, 256],
}

Best Configuration
epochs = 50, batch_size = 64, lr = 0.0005, hidden_dim = 128
```

**Results by Patience**

| Patience | Best Accuracy |
| -------- | ------------- |
| 3        | 76.9 %        |
| 7        | 77.3 %        |


Observation:
Increasing patience from 3 → 7 slightly improved the model’s performance, indicating more stable training and better convergence.

---

### Sentence Transformers


- **Training:** 50 epochs, batch size 32, learning_rate 0.001, patience 3.

##  Grid Search Hyperparameter Tuning (Sentence Transformers)

- Ran **one grid search** (patience = 3) using the parameter grid below, introducing warmup ratio to softly ramp up the learning rate at the beginning to avoid training instability.

**Parameter Grid**
```python
param_grid = {
    "lr": [0.001, 0.0005, 0.002],
    "batch_size": [32, 64],
    "hidden_dim": [128, 256],
    "warmup_ratio": [0.05, 0.1, 0.2]
}

Best Configuration
epochs = 50, batch_size = 64, lr = 0.002, hidden_dim = 256, warmup_ratio = 0.1
```

- **Best Accuracy:** 77.7 %.  



---

## GRU Model Experiments – Summary Table

- Replaced the feedforward architecture with a GRU to better capture sequential dependencies.

| Experiment | Modifications / Features | Patience | Test Accuracy | Notes |
|------------|-------------------------|----------|---------------|-------|
| Initial GRU | Truncated to 300 words, GloVe 300-dim, padding, mean pooling, no bidirectionality | 3 | 76.76% | Overfitting from epoch 1 |
| 1 | Dropout 0.5, Weight decay 1e-5, Bidirectional, Hidden dim 128 → 64 | 5 | 74 – 76.30% | Much less overfitting |
| 2 | Attention pooling (instead of mean) | 5 | 75.31% | – |
| 3a | Truncated to 600 words | 5 | 76.30% | Longer blurbs |
| 3b | Truncated to 150 words | 5 | 75.43% | Shorter blurbs |

- Observation: Using a GRU likely did not better the performance of a simple neural network as most blurbs in the dataset were short.
---

### DistilBERT (Transformers)
  
- **Base model:** `distilbert-base-uncased` (768-dim embeddings, 6 transformer layers, WordPiece vocab ≈30k)  
- **Input processing:** Texts tokenized with HuggingFace `AutoTokenizer`, truncated/padded to **256 tokens** (~200 words) to standardize input lengths for efficient batching  
- **Dataset conversion:** Pandas DataFrames → HuggingFace `Dataset`; tokenized batches converted to PyTorch tensors (`input_ids`, `attention_mask`, `label`)  
- **Training settings:**  
  - Epochs = 3  
  - Learning rate = 2e-5  
  - Weight decay = 0.01  
  - Model was **only trained once** due to long runtime (~3.5 hours)  
- **Fine-tuning:** Classification head (softmax over labels) added for task-specific training


