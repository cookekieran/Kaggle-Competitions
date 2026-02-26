# Predicting Student Test Scores

This folder contains my submission for the Predicting Student Test Scores Kaggle competition: https://www.kaggle.com/competitions/playground-series-s6e1/overview. The goal was to develop a high-performance machine learning pipeline to predict student exam scores based on demographic, behavioral, and academic factors.

My final model achieved:

- **RMSE:** 8.7058  
- **R² Score:** 0.7869  
- **Leaderboard Placement:** 1093rd (Top 23%)

---

## Project Highlights

**Advanced Feature Engineering**
- Developed synthetic features including regression-based weights, polynomial transformations, and interaction terms.

**Data Leakage Mitigation**
- Implemented Out-of-Fold (OOF) target encoding to capture categorical statistics without overfitting.

**Algorithmic Insight**
- Identified and leveraged a synthetic artifact in the dataset (the tenths digit of study hours), significantly improving predictive accuracy.

---

## The Machine Learning Pipeline

### 1. Feature Engineering & Signal Extraction

Beyond the base dataset, several high-impact features were engineered to capture complex relationships:

**Regression-Derived Features**
- Created `reg_feature` and `reg_feature2` using optimal linear weights from the most correlated variables:
  - Study hours  
  - Class attendance  
  - Sleep quality  

**Polynomial Features**
- Modeled the non-linear "two-hump" relationship between study time and exam scores using 3rd-order polynomials.

**Interaction Terms**
- Captured combined behavioral effects such as:
  - `study_hours × sleep_hours`
  - `class_attendance × study_method`

**Synthetic Artifact Capture**
- Discovered that the tenths digit of `study_hours` influenced scores due to the synthetic data generation process.
- Encoded this as `study_hours_digit`.

---

### 2. Advanced Encoding Techniques

**One-Hot Encoding**
- Applied to lower-cardinality categorical features.

**Target Encoding (OOF)**
- Used 5-fold cross-validation to compute:
  - Mean  
  - Standard deviation  
  - Skewness  
- Prevented data leakage while providing powerful statistical priors to the model.

---

### 3. Model Architecture

The core model is a highly tuned **XGBoost Regressor**.

**Configuration**
- 1000 estimators  
- Learning rate: 0.05  
- Max depth: 6  

**Validation Strategy**
- Early stopping with 50 rounds  
- Ensured strong generalization and reduced overfitting  

---

## Results and Impact

Feature importance analysis revealed that:

- Class Attendance  
- Engineered Study Hours Digit  
- Regression-Based Features  

were the strongest drivers of predictive performance.

| Metric | Score |
|--------|--------|
| RMSE | 8.7058 |
| R² Score | 0.7869 |

---

## Next Steps

**Ensemble Modeling**
- Integrate LightGBM and CatBoost using a Voting Regressor.

**Hyperparameter Optimisation**
- Implement GridSearchCV for parameter optimisation

**Feature Engineering**
- Investigation into digits of features to take advantage of the synthetic data