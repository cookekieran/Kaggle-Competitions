# Titanic: Machine Learning from Disaster


This folder contains my solution for the Titanic Kaggle competition. By leveraging advanced feature engineering and ensemble methods, this notebook achieved a final standing of 1,624th out of 11,654 participants, placing in the top 14% at the time of writing. 

---

## Project Overview

The goal is to build a predictive model that answers the question:

**"What sorts of people were more likely to survive?"**

The model uses structured passenger data, including:

- Name  
- Age  
- Gender  
- Socio-economic class  
- Ticket information  
- Cabin data  
- Family relationships  

---

## Data Insights

Initial exploratory data analysis revealed several important patterns that guided feature engineering:

**Survival Disparities**
- Females had a survival rate of approximately **74%**
- Males had a survival rate of approximately **19%**

**Socio-Economic Impact**
- 1st Class passengers: approximately **63% survival rate**
- 3rd Class passengers: approximately **24% survival rate**

**Missing Data**
- Age: approximately **20% missing**
- Cabin: approximately **77% missing**
- Embarked: approximately **0.22% missing**

---

## Feature Engineering

To improve predictive performance, several engineered features were introduced:

### Family Size Grouping

- Combined `SibSp` and `Parch` to create a `family_size` feature.
- Grouped into:
  - Alone  
  - Small  
  - Medium  
  - Large  
- "Small" families (2–4 members) showed the highest survival rates (approximately **58%**).

### Title Extraction

- Extracted titles (Mr, Miss, Mrs, etc.) from the `Name` field.
- Rare titles were grouped into broader categories such as:
  - Military  
  - Noble  
- These categories showed strong correlation with survival outcomes.

### Ticket Analysis

- Extracted ticket prefixes and ticket frequency counts.
- Identified groups traveling together.
- Higher ticket frequency often correlated with specific survival patterns.

### Cabin Binary Mapping

- Due to high missingness, `Cabin` was simplified into:
  - `cabin_assigned = 1` if cabin was present  
  - `cabin_assigned = 0` if missing  
- Passengers with assigned cabins had a **66.7% survival rate**.

---

## Machine Learning Pipeline

The project leverages **scikit-learn Pipelines** to ensure clean, reproducible preprocessing and modeling.

### Preprocessing

**Imputation**
- Median imputation for `Age` and `Fare`
- Most-frequent imputation for categorical variables

**Encoding**
- `OneHotEncoder` for nominal features:
  - Sex  
  - Title  
  - Embarked  
- `OrdinalEncoder` for grouped features:
  - Family size categories  

---

## Models Evaluated

- Random Forest Classifier  
- Decision Tree Classifier  
- XGBoost  
- Support Vector Classifier (SVC)  
- Voting Classifier (ensemble)  

---

## Next Steps


### 1. Further Feature Engineering

**Family Survival Rates**

- Create a **Family Survival Group** feature, extracting surnames to identify family groupings.
- Calculate whether other members of a passenger's family survived, which would serve as a strong predictive signal, as families often survived or perished together.

**Deck Extraction**

- Extract the deck letter (A, B, C, etc.) from the `Cabin` variable. Deck location acts as a proxy for proximity to lifeboats. Provides additional spatial context related to survival probability.

---

### 2. Model Stacking & Blending

Instead of relying solely on a `VotingClassifier`, implement a **Stacked Generalization** approach:

- Train base models:
  - XGBoost  
  - Random Forest  
  - Support Vector Classifier (SVC)  

- Generate out-of-fold predictions from each base model.
- Train a **Meta-Learner** (e.g., Logistic Regression) on these predictions.
- The meta-model learns optimal weighting of base model outputs for improved performance.

---

### 3. Handling Data Leakage

Improve model robustness and evaluation reliability by:

- Implementing **Repeated Stratified K-Fold Cross-Validation**
- Ensuring strict separation between training and validation folds
- Reducing variance in performance estimates
- Minimizing the risk of data leakage