# 📘 Machine Unlearning for HR Attrition Prediction

A complete implementation of Machine Learning, Deep Learning, and Machine Unlearning techniques using the **IBM HR Analytics Attrition** dataset.

---

## 🔍 Overview

This project shows how ML and DL models can be trained, evaluated, and then **selectively unlearned** (i.e., remove influence of specific training samples) to meet:

- Privacy regulations (GDPR, CCPA, DPDP Act)  
- The “Right to be Forgotten”  
- Ethical and secure AI requirements  
- Enterprise compliance for sensitive HR data

It demonstrates both **approximate** and **exact** machine unlearning techniques applied to classical ML models, ANN, 1D-CNN, and the **SISA** framework.

---

## 🚀 Key Features

### ✅ Complete Data Pipeline
- Data validation & integrity checks  
- One-hot encoding & label encoding  
- Standardization of numerical variables  
- SMOTE for class imbalance correction (applied to training set only)  
- Stratified train–test split (80 / 20)

### 🤖 Machine Learning Models
- **Logistic Regression**  
- **Random Forest**  
- **XGBoost**

### 🧠 Deep Learning Models
- **ANN** with Dropout & EarlyStopping  
- **1D-CNN** adapted for tabular data

### 🧹 Machine Unlearning Techniques

**Approximate Unlearning**
- Amnesiac Retraining (retrain on dataset minus forget set)  
- Weight Masking (zero out small-magnitude weights)  
- Gradient Reversal / Gradient Ascent (update weights to reduce influence of forget set)

**Exact Unlearning**
- **SISA (Sharded • Isolated • Sliced • Aggregated)**  
  - Partition training data into shards and slices  
  - Retrain only affected slice(s) when deletion is requested  
  - Aggregate shard models to obtain final model

### 📊 Full Evaluation Suite
- Accuracy, Precision, Recall, F1-Score, ROC–AUC  
- Forgetting metric (e.g., Influence loss or Forgetfulness Rate)  
- Membership-inference and reconstruction tests (privacy leakage)  
- Runtime / compute cost and audit logs comparison  
- Pre- vs Post-unlearning performance

---

## 📂 Project Structure

