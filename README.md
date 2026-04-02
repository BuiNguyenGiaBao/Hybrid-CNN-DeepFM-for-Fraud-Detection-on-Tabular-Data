# Hybrid CNN–DeepFM for Fraud Detection on Tabular Data

## Abstract
This project proposes a Hybrid CNN–DeepFM architecture for fraud detection using the IEEE-CIS dataset. The model integrates convolutional feature extraction, attention mechanisms, and feature interaction learning to handle high-dimensional tabular data. The final model achieves high precision (~0.74) with moderate discriminative performance (ROC-AUC ~0.80), though recall remains limited (~0.11) due to conservative decision behavior.
=======

Fraud detection on large-scale tabular data is challenging because of extreme class imbalance, high dimensionality, and complex feature interactions. This project proposes a **Hybrid CNN–DeepFM architecture** for fraud detection on the **IEEE-CIS Fraud Detection Dataset**. The model combines CNN-based representation learning, attention pooling, low-rank bilinear interaction modeling, and DeepFM-based higher-order feature learning.

To improve minority-class detection, the training pipeline is further enhanced with **Focal Loss**, **Weighted Random Sampling**, and **validation-based threshold tuning**. Experimental results show that the optimized pipeline substantially improves fraud recall while maintaining strong discriminative ability in terms of ROC-AUC.

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

### 4.1 CNN-Based Feature Extractor


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

### 4.2 Low-Rank Bilinear Interaction Modeling

After attention pooling, the model applies **low-rank bilinear pooling** to capture pairwise interactions among latent features. This helps represent second-order dependencies efficiently without a full bilinear parameter explosion.

### 4.3 DeepFM for Relationship Learning

The learned CNN embeddings are passed into a **DeepFM** module, which includes:

- **Linear part** for first-order effects
- **Factorization Machine part** for second-order interactions
- **Deep neural network part** for higher-order nonlinear relationships

This architecture enables the model to learn complex feature interactions without manual feature crossing.

---

## 5. Imbalance-Aware Optimization

Because fraud detection is highly imbalanced, the optimized pipeline includes several additional techniques:

### 5.1 Focal Loss
Focal Loss is used to place greater emphasis on hard-to-classify minority fraud samples.

### 5.2 Weighted Random Sampling
A `WeightedRandomSampler` is applied to the training loader so that fraud samples are observed more frequently during training.

### 5.3 Threshold Tuning
Instead of using a fixed threshold of 0.5, the classification threshold is tuned on the validation set. This makes the pipeline more suitable for **recall-oriented fraud detection**.

---

## 6. Training Setup

- **Optimizer**: AdamW
- **Learning rate scheduler**: ReduceLROnPlateau
- **Loss**: Focal Loss (optimized version)
- **Gradient clipping**: enabled
- **Early stopping / model selection**: based on validation fraud-oriented metric
- **Evaluation metrics**:
  - ROC-AUC
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix

Accuracy is reported for completeness, but it is not the main focus because of class imbalance.

---

## 7. Experimental Results

### Main observations

The original pipeline achieved strong ROC-AUC but relatively low fraud recall. After introducing imbalance-aware optimization, the model significantly improved fraud detection recall.

### Representative result of the optimized pipeline

- **Fraud Precision**: 0.50
- **Fraud Recall**: 0.66
- **Fraud F1-score**: 0.57
- **Overall Accuracy**: 0.96

### Interpretation

This result indicates a more recall-oriented fraud detection behavior:

- The optimized pipeline detects substantially more fraudulent transactions than the earlier conservative version.
- The trade-off is a reduction in fraud precision, meaning more false positives are produced.
- Such a trade-off is often acceptable in fraud detection scenarios where missing fraudulent transactions is more costly than triggering additional alerts.

---

## 8. Discussion

The proposed Hybrid CNN–DeepFM architecture is effective at learning **intra-instance feature interactions** in tabular transaction data. CNN, attention pooling, bilinear interaction, and DeepFM together provide a strong mechanism for latent pattern extraction and high-order relationship learning.

However, the model still operates on **independent tabular records**. Therefore, it mainly learns relationships **within each transaction**, rather than explicit relationships **between transactions**.

This is an important limitation for fraud detection, because fraudulent behavior often emerges through:
- shared devices
- shared cards
- shared addresses
- repeated identity patterns
- temporal interaction networks

Thus, while the optimized pipeline improves recall substantially, it is still constrained by the absence of explicit graph-based relational modeling.

---

## 9. Limitations

This project has several limitations:

1. The model does not explicitly model inter-transaction graph structure.
2. Fraud precision decreases when recall is aggressively optimized.
3. Performance still depends heavily on preprocessing quality and available features.
4. The current pipeline is better at **feature interaction learning** than true **graph relational learning**.

---

## 10. Future Work

Possible extensions include:

- Constructing graph-based transaction relationships
- Comparing the current pipeline with **Graph Neural Networks (GNNs)**
- Incorporating temporal sequence modeling
- Combining the model with gradient boosting or ensemble strategies
- Improving feature engineering for card, device, address, and time-based behavioral patterns

---

## 11. Conclusion

This project demonstrates that a **Hybrid CNN–DeepFM** architecture can effectively model complex feature interactions for tabular fraud detection. The optimized training pipeline, enhanced with **Focal Loss**, **Weighted Random Sampling**, and **threshold tuning**, significantly improves fraud recall on the IEEE-CIS Fraud Detection dataset.

Although the model remains limited compared with explicit relational or graph-based approaches, it provides a strong deep learning framework for fraud detection on structured tabular data.

---

## Project Structure

```bash
project/
│
├── data/
│   └── merge/
│       ├── train_processed.csv
│       ├── test_processed.csv
│       └── val_processed.csv
│
├── modules/
│   ├── cnn_for_extract_feature.py
│   ├── deepfm_for_relationship.py
│   └── training.py
│
├── results/
│   ├── classification_report.txt
│   ├── evaluation_metrics.png
│   └── training_history.png
│
├── best_fraud_model.pth
└── README.md
```

---

## How to Run

### 1. Prepare the environment

Install the required Python packages before running the project:

```bash
pip install torch torchvision torchaudio
pip install pandas numpy scikit-learn matplotlib seaborn tqdm
```

### 2. Prepare the data

Make sure the processed dataset files are stored in the following location:

```bash
data/merge/train_processed.csv
data/merge/test_processed.csv
```

If you also have a validation file, it can be stored as:

```bash
data/merge/val_processed.csv
```

### 3. Train the model

Run the training pipeline with focal loss and fraud-oriented threshold tuning:

```bash
python modules/training.py --mode train --use_focal_loss
```

You may also explicitly set the main hyperparameters:

```bash
python modules/training.py --mode train --use_focal_loss --focal_alpha 0.90 --focal_gamma 3.0 --threshold_metric recall
```

### 4. Train and generate predictions

To train the model and then generate predictions on the processed test set:

```bash
python modules/training.py --mode train_and_predict --use_focal_loss
```

### 5. Predict using a saved checkpoint

To load a trained model checkpoint and run prediction only:

```bash
python modules/training.py --mode predict --checkpoint best_fraud_model.pth
```

### 6. Output files

After training, the following outputs will typically be generated:

- `best_fraud_model.pth` – saved model checkpoint
- `results/classification_report.txt` – classification report
- `results/evaluation_metrics.png` – confusion matrix and ROC curve
- `results/training_history.png` – training and validation curves
- `submission.csv` – fraud probability predictions for the test set

### 7. Notes

- The optimized pipeline uses **Focal Loss**, **WeightedRandomSampler**, and **threshold tuning** to improve fraud recall.
- The default processed file paths are resolved automatically from the project structure.
- GPU is used automatically if CUDA is available; otherwise, the code runs on CPU.

---

## References

1. Guo, H., Tang, R., Ye, Y., Li, Z., & He, X. (2017). **DeepFM: A Factorization-Machine Based Neural Network for CTR Prediction.** *Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence (IJCAI 2017)*, 1725–1731.

2. Kaggle. (2019). **IEEE-CIS Fraud Detection.** Kaggle Competition Dataset.

3. Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). **Focal Loss for Dense Object Detection.** *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 2980–2988.

4. Rendle, S. (2010). **Factorization Machines.** *2010 IEEE International Conference on Data Mining*, 995–1000.

5. Somepalli, G., Goldblum, M., Schwarzschild, A., Bruss, C. B., & Goldstein, T. (2021). **SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training.** *arXiv preprint arXiv:2106.01342*.

6. Borisov, V., Leemann, T., Seßler, K., Haug, J., Pawelczyk, M., & Kasneci, G. (2022). **Deep Neural Networks and Tabular Data: A Survey.** *IEEE Transactions on Neural Networks and Learning Systems*.

7. Shwartz-Ziv, R., & Armon, A. (2022). **Tabular Data: Deep Learning is Not All You Need.** *Information Fusion*, 81, 84–90.

8. IEEE Computational Intelligence Society. **IEEE-CIS Fraud Detection Dataset Documentation.**

