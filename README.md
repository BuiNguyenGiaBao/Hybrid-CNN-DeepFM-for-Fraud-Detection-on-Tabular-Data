# Hybrid CNN–DeepFM for Fraud Detection on Tabular Data

## Abstract
This project proposes a Hybrid CNN–DeepFM architecture for fraud detection using the IEEE-CIS dataset. The model integrates convolutional feature extraction, attention mechanisms, and feature interaction learning to handle high-dimensional tabular data. The final model achieves high precision (~0.74) with moderate discriminative performance (ROC-AUC ~0.80), though recall remains limited (~0.11) due to conservative decision behavior.

---

## 1. Introduction
Fraud detection is a challenging task due to extreme class imbalance and complex feature interactions. Traditional machine learning models often struggle to capture nonlinear dependencies in tabular data. This project explores a deep learning approach combining CNN and DeepFM to improve representation learning and predictive performance.

---

## 2. Dataset
The IEEE-CIS Fraud Detection dataset contains transaction and identity features derived from real-world e-commerce data. Key characteristics include:
- Severe class imbalance
- High dimensionality
- Mixed feature types (categorical and numerical)

---

## 3. Methodology
The proposed model consists of:
- CNN-based feature extractor for local pattern learning
- Attention pooling for feature importance weighting
- Bilinear interaction module for feature crossing
- DeepFM component for implicit and explicit interactions

Additionally, imbalance handling techniques are applied:
- Focal Loss
- Weighted Random Sampling
- Threshold tuning

---

## 4. Experimental Results

### Final Performance:
- Precision: ~0.74
- Recall: ~0.11
- F1-score: ~0.19
- ROC-AUC: ~0.80
- PR-AUC: ~0.30

### Confusion Matrix Insight:
- True Positives: 460
- False Negatives: 3673
- False Positives: 159

The model demonstrates high reliability in predicted fraud cases but misses a significant portion of actual fraud instances.

---

## 5. Discussion
The model exhibits a high-precision, low-recall trade-off. This indicates a conservative prediction strategy, where fraud is only detected when the model is highly confident. While this reduces false positives, it leads to lower coverage of fraudulent transactions.

---

## 6. Conclusion
The Hybrid CNN–DeepFM model effectively captures feature interactions in tabular fraud detection tasks. It is particularly suitable for high-precision scenarios, such as fraud verification systems. However, improvements are needed to increase recall for broader fraud coverage.

---

## 7. Future Work
- Improve recall using graph-based models (GNN)
- Incorporate temporal transaction patterns
- Ensemble with boosting-based models (e.g., LightGBM, XGBoost)
- Dynamic threshold optimization
