This folder contains my attempt at the titanic kaggle competition: https://www.kaggle.com/competitions/titanic. 

My work achieved a submission accuracy of 0.78468 (submission11).

### Model Features

The following table outlines the features used in the model, distinguishing between raw data extracted directly from the Titanic dataset and features that were engineered:

| Feature | Type | Description |
| :--- | :--- | :--- |
| **Sex** | Raw | Original gender column from the dataset. |
| **Embarked** | Raw | Port of embarkation (C, Q, or S). |
| **Pclass** | Raw | Ticket class (1, 2, or 3). |
| **Fare** | Raw | The price paid for the ticket. |
| **family_size_grouped** | Engineered | Created by combining `SibSp` and `Parch` and grouping them into categories. |
| **title** | Engineered | Extracted from the `Name` column (e.g., Mr, Mrs, Miss, Master). |
| **name_length** | Engineered | A numerical feature representing the character count of the passenger's name. |
| **ticket_number_count** | Engineered | Created by counting how many passengers shared the same ticket number. |
| **cabin_assigned** | Engineered | A binary feature (1/0) indicating whether a cabin number was recorded. |

## Model Selection

Twelve different approaches using Scikit-Learn and XGBoost pipelines were evaluated for this project. This allowed for a comparison across various algorithms, ranging from simple probabilistic models to complex ensemble methods.

| Model Variable | Algorithm | Category |
| :--- | :--- | :--- |
| `Y_pred` | **Random Forest** | Ensemble (Bagging) |
| `Y_pred2` | **Decision Tree** | Tree-based |
| `Y_pred3` | **K-Nearest Neighbors** | Instance-based |
| `Y_pred4` | **Support Vector Classifier** | Kernel-based |
| `Y_pred5` | **Logistic Regression** | Linear Model |
| `Y_pred6` | **Gaussian Naive Bayes** | Probabilistic |
| `Y_pred7` | **XGBoost** | Ensemble (Boosting) |
| `Y_pred8` | **AdaBoost** | Ensemble (Boosting) |
| `Y_pred9` | **Extra Trees** | Ensemble (Bagging) |
| `Y_pred10` | **Gradient Boosting** | Ensemble (Boosting) |
| `Y_pred11` | **Voting Ensemble 1** | Meta-Estimator |
| `Y_pred12` | **Voting Ensemble 2** | Meta-Estimator |

The use of `pipelines` ensured that all preprocessing steps (scaling, encoding, and imputation) were applied identically to the test set for every model, preventing data leakage and ensuring reproducible results.

### Next steps

Currently, missing values are handled with SimpleImputer strategies, replacing null data with mode or median values. A more complex approach, such as using Iterative Imputer or KNN Imputer, could provide more realistic estimates for missing age and cabin data.

Additionally, engineered features involves grouping. A more advanced feature engineering approach may uncover further relationships within the raw features. For example, interaction terms between Age and class or title and fare may better segment the passenger list. The Cabin feature is sparse, but the letter prefix indicates the deck and therefore the proximity to lifeboats, which is a factor not yet considered.

