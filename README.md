# PriceFusion - ZeroVector
**Amazon ML Challenge 2025 | Smart Product Pricing Solution**

**Team:** ZeroVector  
**Members:** Jenil Savaj, Ravi Pratap, Siddharth Thakur  
**Submission Date:** 13 October 2025  

---

## 🧠 Executive Summary
PriceFusion predicts optimal product prices using a **multimodal learning framework** that fuses textual, visual, and handcrafted features.  
The model leverages **transformer-based embeddings** for product descriptions and images, integrated via **LightGBM** for efficient and interpretable predictions.  
Our final solution achieved a **validation SMAPE of 37.85%**, demonstrating robustness and balanced performance.

---

## ⚙️ Methodology Overview

### 🔍 Problem Analysis
We analysed catalog content and product images to capture the key factors influencing pricing.  
EDA revealed strong relationships between descriptive richness, product quantity, and price variability.  
High-quality images correlated with higher price confidence.

**Key Observations**
- Text length and presence of numerical keywords affected pricing patterns.  
- Product quantity showed non-linear relations with price.  
- Image clarity correlated with price confidence.

### 🧩 Solution Strategy
**Approach Type:** Hybrid Multimodal Model  
**Core Innovation:** Unified representation of text and image embeddings using transformer models to improve semantic understanding and pricing accuracy.  

---

[Text Input] → [SentenceTransformer Encoder]
↓
[Image Input] → [CLIP Vision Encoder]
↓
[Feature Concatenation] → [LightGBM Regressor] → [Predicted Price]


---

## 🔡 Model Components

### 1. Text Processing Pipeline
- Preprocessing: Lowercasing, punctuation removal, text cleaning  
- Model: `all-mpnet-base-v2` (SentenceTransformer)  
- Embedding Dimension: 768  
- Batch Size: 64  

### 2. Image Processing Pipeline
- Preprocessing: RGB conversion, resizing to 224×224, normalization  
- Model: `CLIP ViT-B/32` (Vision Encoder)  
- Embedding Dimension: 512  
- Batch Size: 48  

### 3. Feature Fusion & Regression
Combined embeddings and handcrafted features were fed into a **LightGBM Regressor** trained with early stopping and feature-wise optimization.

---

## 📊 Model Performance

| Metric | Score |
|:-------|:------:|
| **SMAPE** | 37.85% |
| **MAE (log-space)** | 0.36978 |
| **MSE (log-space)** | 0.275008 |

The model balanced interpretability and accuracy, performing robustly across diverse product categories and noise levels in multimodal data.

---

## 🚀 Future Improvements
- Fine-tuning CLIP for domain-specific product imagery  
- Dynamic feature weighting during fusion  
- Integration of category-level priors for better contextual pricing  

---

## 💻 Tech Stack
- Python  
- PyTorch  
- Hugging Face Transformers  
- OpenAI CLIP  
- LightGBM  
- NumPy, Pandas, Scikit-learn  

---

## 🏁 Conclusion
By combining **semantic understanding from text**, **visual cues from images**, and **quantitative features**,  
PriceFusion effectively captures the multi-faceted aspects of product pricing.  
The approach highlights the potential of **multimodal fusion** for real-world e-commerce applications.

---



