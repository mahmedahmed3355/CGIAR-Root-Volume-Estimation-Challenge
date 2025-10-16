# 🌱 CGIAR Root Volume Estimation Challenge

### 🧠 Overview

This project was developed as part of the **CGIAR Root Volume Estimation Challenge**, which aims to accurately estimate the **root volume** of plants from images using advanced machine learning and computer vision techniques.
The task involves building a regression model capable of predicting the continuous target variable — *root volume* — from multi-modal data sources, including image-based and tabular features.

---

## 🚀 Project Highlights

* **Competition:** [CGIAR Root Volume Estimation Challenge](https://www.zindi.jp/competitions/)
* **Task Type:** Regression
* **Frameworks Used:** PyTorch, LightGBM, CatBoost, XGBoost, scikit-learn, timm
* **Public Score:** 1.06
* **Private Score:** 1.38
* **Author:** Mohamed Ahmed (`mohamed2236945`)

---

## 📂 Project Structure

```
CGIAR-Root-Volume-Estimation/
│
├── data/
│   ├── train/                # Training images
│   ├── test/                 # Test images
│   ├── train.csv             # Metadata and root volume labels
│   └── sample_submission.csv
│
├── src/
│   ├── preprocess.py         # Image and feature preprocessing
│   ├── dataset.py            # Custom PyTorch dataset class
│   ├── feature_engineering.py# Feature generation and augmentation
│   ├── model_lightgbm.py     # LightGBM training script
│   ├── model_catboost.py     # CatBoost training script
│   ├── model_xgb.py          # XGBoost training script
│   ├── model_mlp.py          # Simple PyTorch MLP (optional)
│   └── utils.py              # Helper functions
│
├── notebooks/
│   └── exploration.ipynb     # EDA and visualization notebook
│
├── outputs/
│   ├── submissions/          # Final submission CSV files
│   └── models/               # Saved trained models (.pkl / .bin)
│
├── requirements.txt
├── README.md
└── main.py                   # Main training & inference pipeline
```

---

## ⚙️ Installation

```bash
git clone https://github.com/mahmedahmed3355/CGIAR-Root-Volume-Estimation.git
cd CGIAR-Root-Volume-Estimation
pip install -r requirements.txt
```

### Requirements

* Python 3.10+
* PyTorch
* timm
* scikit-learn
* xgboost
* lightgbm
* catboost
* Pillow
* numpy, pandas, matplotlib, seaborn

---

## 🧩 Approach

### 1. **Data Preprocessing**

* Converted and resized all input images for consistent shape and normalization.
* Augmented images using standard geometric transformations (rotation, flip, color jitter).
* Extracted **image embeddings** using pretrained `timm` models (e.g. EfficientNet, ResNet).
* Combined image features with **tabular metadata** to form a hybrid feature vector.

### 2. **Feature Engineering**

* Statistical aggregations and normalization of numerical features.
* Polynomial feature combinations for selected predictors.
* Added domain-specific ratios and interaction terms.
* Applied PCA for dimensionality reduction and noise filtering.

### 3. **Modeling**

Trained multiple regressors:

* 🟢 **CatBoostRegressor**
* 🔵 **LightGBMRegressor**
* 🔴 **XGBoostRegressor**
* (Optional) PyTorch **MLPRegressor**

Each model was fine-tuned using Bayesian optimization or grid search on validation folds.

### 4. **Ensembling**

* Averaged predictions from top-performing models using weighted blending:

  ```python
  final_pred = 0.4 * catboost_pred + 0.35 * lgb_pred + 0.25 * xgb_pred
  ```
* Ensemble reduced overfitting and improved generalization on the private leaderboard.

---

## 📈 Results

| Model                | Public Score | Private Score |
| -------------------- | ------------ | ------------- |
| LightGBM             | 1.09         | 1.42          |
| CatBoost             | 1.07         | 1.40          |
| XGBoost              | 1.08         | 1.41          |
| **Ensemble (Final)** | **1.06**     | **1.38**      |

---

## 💾 Inference

```bash
python main.py --mode inference --model ensemble
```

This script loads trained models and generates a submission file:

```
submissions/final_submission.csv
```

---

## 🧰 Key Learnings

* Hybrid feature fusion between image embeddings and tabular data can greatly improve accuracy.
* Ensemble averaging across gradient boosting models is highly effective for small data challenges.
* Proper validation and cross-checking prevent leaderboard overfitting.

---

## 📜 License

This project is released under the **MIT License**.

---

## 👤 Author

**Mohamed Ahmed**
AI Engineer | ML / CV / LLM Specialist
📧 [engmohamedelshrbeny@gmail.com](mailto:engmohamedelshrbeny@gmail.com)
🌍 [LinkedIn](https://linkedin.com/in/engmohamedelshrbeny) | [GitHub](https://github.com/engmohamedelshrbeny)
