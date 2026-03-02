# Heart Disease Prediction – Competition Solution

This repository contains my solution for the Heart Disease Prediction competition.

The goal was to predict the presence or absence of heart disease based on clinical parameters.

**Final Results:**
- **Training Accuracy:** 89.20%
- **Rank:** 2246 out of 4507 participants

---

## Project Overview

This project is a binary classification task.

The pipeline includes:
- Extensive feature engineering
- Target encoding with cross-validation to prevent data leakage
- Hyperparameter optimization for ensemble models

---

## Key Techniques & Features

### 1. Advanced Feature Engineering

#### Binning & Grouping
- Applied `pd.qcut` and `pd.cut` to continuous variables such as Max HR and Cholesterol
- Created categorical groups to help capture non-linear relationships

#### Regression-Based Features
- Used a `LinearRegression` model to generate a new feature called `reg_feature`
- This acts as a meta-feature representing a linear combination of key predictors such as:
  - Thallium
  - Chest pain type

#### Interaction Terms
Created interaction features to capture combined effects between demographic and clinical variables, such as:
- `Male_x_Thallium_7`
- `Age_x_Chest_pain_type_4`

---

### 2. Target Encoding

- Implemented target encoding for categorical variables
- Used 5-fold cross-validation within the training set
- Prevented data leakage and reduced overfitting

---

### 3. Model Selection & Tuning

#### Random Forest
- Tuned using `RandomizedSearchCV`
- Best parameters:
  - `n_estimators: 300`
  - `max_depth: 10`
  - `max_samples: 0.8`
- **Training Accuracy:** 88.92%

#### XGBoost (Final Model)
- Used `XGBClassifier`
- Key parameters:
  - `learning_rate: 0.1`
  - `colsample_bytree: 0.3`
- **Training Accuracy:** 89.20%

---

## Performance Summary

| Model              | Training Accuracy |
|--------------------|-------------------|
| Random Forest      | 88.92%            |
| XGBoost (Final)    | 89.20%            |