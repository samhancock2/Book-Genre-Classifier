# 🧠 Text Classification with Traditional and Transformer-Based Embeddings

## 📘 Project Overview
This project benchmarks multiple natural language processing (NLP) approaches for **text classification** on a dataset of ~34 k items (product or text descriptions).  
The objective was to compare traditional embedding-based and sequence models against modern transformer-based architectures.

Each model was trained on the same preprocessed dataset and evaluated on test accuracy.

---

## 🧾 Model Performance Summary

| Model Type | Configuration Highlights | Test Accuracy |
|-------------|--------------------------|----------------|
| **GloVe 50D** | 50-dimensional GloVe (6B) | **71 %** |
| **GloVe 100D** | 100-dimensional GloVe (6B) | **72 %** |
| **GloVe 200D** | 200-dimensional GloVe (6B) | **75 %** |
| **GloVe 300D** | 300-dimensional GloVe (6B) | **76 %** |
| **Sentence‑Transformer** | 384-dimensional sentence vectors (`all‑MiniLM‑L6‑v2`) | **76 %** |
| **GRU (optimized)** | Bi‑GRU + dropout + attention, 300D GloVe inputs | **0.74 – 0.77** |
| **DistilBERT (Base)** | Transformer fine‑tuned on task | **82.2 % ⭐ (best)** |

---

## ⚙️ Experimental Details

### 🔹 GloVe Models
- Used pre-trained **GloVe 6B** embeddings of 50–300 dimensions.  
- Classifier: feed-forward network (hidden_dim = 128).  
- **Training:** 50 epochs, batch size 32, learning_rate 0.001, patience 3.  
- Performance improved steadily with embedding size (plateaued around 300 D).

**Hyperparameter Search**
- Ran grid search twice:  
  - Patience = 3 → best 76.9 %  
  - Patience = 7 → best 77.3 %  
- Best config: `epochs = 50`, `batch = 64`, `lr = 5e‑4`, `hidden_dim = 128`.

---

### 🔹 Sentence‑Transformer Models
- Base model: **`sentence-transformers/all‑MiniLM‑L6‑v2`** (384-dimensional sentence vectors).  
- **Training:** epochs = 50, batch size 64, learning_rate = 0.002, hidden_dim = 256, warmup ratio = 0.1, patience = 3.  
- **Accuracy:** 77.7 %.  
- Provided strong semantic embeddings but slightly below transformer fine‑tuning.

---

### 🔹 GRU Models
**Initial setup**
- Truncated texts to 300 words per example.  
- Input embeddings: GloVe 300D.  
- Padding + mean pooling; non‑bidirectional GRU; patience = 3.  
- Test accuracy = 0.7676 with early overfitting.

**Further Experiments**
1. Dropout 0.5 + weight_decay 1e‑5 + bidirectional GRU + hidden_dim 128 → 64 + patience = 5  
   → Reduced overfitting, accuracy ≈ 0.74 – 0.76.  
2. Attention pooling instead of mean → accuracy ≈ 0.753.  
3. Sequence length tests (150 / 300 / 600 tokens):  
   - 600 = 0.763, 150 = 0.754.

---

### 🔹 DistilBERT (Transformers)
**Model Configuration**
- Base model: `distilbert-base-uncased`.  
- Embedding dim = 768, 6 transformer layers, WordPiece vocab ≈ 30 k.  
- Max sequence length = 256 tokens (≈ 200 words).  
- Fine‑tuned with classification head (softmax over labels).  

**Training Arguments**