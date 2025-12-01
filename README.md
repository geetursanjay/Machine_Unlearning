📘 Machine Unlearning for HR Attrition Prediction

A complete implementation of Machine Learning, Deep Learning, and Machine Unlearning techniques using the IBM HR Analytics Attrition Dataset.

🔍 Overview

This project explores how modern machine learning and deep learning models can be trained, evaluated, and then selectively unlearn specific training samples to meet:

Privacy regulations (GDPR, CCPA, DPDP Act)

The “Right to be Forgotten”

Ethical and secure AI requirements

Enterprise compliance for sensitive HR data

The repository demonstrates approximate and exact machine unlearning techniques implemented over classical ML models, ANN, 1D-CNN, and the SISA framework.

🚀 Key Features

Complete Data Pipeline

Data validation & cleaning

One-hot encoding, label encoding

Standardization of numerical features

SMOTE-based class balancing

Stratified train-test split

Machine Learning Models

Logistic Regression

Random Forest

XGBoost

Deep Learning Models

ANN with Dropout + EarlyStopping

1D-CNN adapted for tabular data

Machine Unlearning Techniques

Approximate Unlearning

Amnesiac Retraining

Weight Masking

Gradient Reversal / Gradient Ascent

Exact Unlearning

SISA Framework (Sharded • Isolated • Sliced • Aggregated)

Full Evaluation Suite
Accuracy, Precision, Recall, F1, AUC, Forgetting score, membership inference robustness, and post-unlearning performance comparison.
