# ❤️ Heart Disease Prediction

<p align="center">

  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">

  <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn">

  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">

  <img src="https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">

  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">

</p>

<p align="center">
  <b>End-to-End Machine Learning Pipeline for Heart Disease Classification</b>
</p>

<p align="center">
  Data Preprocessing • EDA • Feature Engineering • Model Training • GridSearchCV • Model Evaluation
</p>

---

## 📌 Overview

**Heart Disease Prediction** is an end-to-end machine learning project designed to classify whether a patient is likely to belong to the **heart-disease** or **non-heart-disease** class based on clinical and demographic features.

The project implements a structured machine learning workflow covering:

- Data preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Feature scaling
- Multiple classification algorithms
- Hyperparameter optimization using `GridSearchCV`
- Cross-validation
- Model evaluation
- Comparative model analysis

Six classical machine learning algorithms were evaluated:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors
4. Support Vector Machine
5. Random Forest
6. AdaBoost

---

## 🎯 Project Objectives

The primary objectives are to:

- Build a reproducible machine learning pipeline for binary classification.
- Analyze clinical features associated with the target variable.
- Compare multiple supervised learning algorithms.
- Optimize model hyperparameters using cross-validation.
- Evaluate models using accuracy, precision, recall, and F1-score.
- Analyze the generalization performance of the trained models.
- Provide a clean foundation for future model deployment.

---

# 🧠 Machine Learning Pipeline

```text
                    ┌─────────────────────┐
                    │      Raw Dataset    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Data Validation     │
                    │ & Cleaning          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Exploratory Data    │
                    │ Analysis            │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Feature Engineering │
                    │ & Encoding          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Feature Scaling     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Train / Test Split  │
                    └──────────┬──────────┘
                               │
                               ▼
              ┌──────────────────────────────────┐
              │       Model Development          │
              │                                  │
              │ Logistic Regression              │
              │ Decision Tree                    │
              │ KNN                              │
              │ SVM                              │
              │ Random Forest                    │
              │ AdaBoost                         │
              └───────────────┬──────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │   GridSearchCV      │
                    │ Hyperparameter      │
                    │ Optimization        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Cross Validation    │
                    │ & Evaluation        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Model Comparison    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Final Prediction    │
                    └─────────────────────┘
