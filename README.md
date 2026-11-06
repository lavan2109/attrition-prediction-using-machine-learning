# Employee Attrition Prediction to Support HR Decision-Making Using Machine Learning

A machine learning project that predicts which employees are likely to leave an organization, so HR can take proactive retention actions.

## Problem Statement
Employee attrition is costly: recruitment, training and lost productivity. This project builds a classification model that identifies employees at risk of leaving and explains the key factors driving attrition.

## Dataset
IBM HR Analytics Employee Attrition & Performance (Kaggle)
- 1,470 employees, 35 features
- Target: `Attrition` (Yes/No), about 16% positive class (imbalanced)

File used: `WA_Fn-UseC_-HR-Employee-Attrition.csv`

## Methodology
1. Data cleaning (removed constant and ID columns)
2. Exploratory data analysis
3. Feature engineering (income per year of experience, promotion ratio, average tenure, satisfaction score)
4. Preprocessing pipeline (StandardScaler + OneHotEncoder)
5. Model comparison: Logistic Regression, Random Forest, Gradient Boosting (5-fold stratified CV)
6. Hyperparameter tuning (GridSearchCV)
7. Threshold tuning using F2 score to prioritise recall on employees who leave
8. Explainability using permutation importance
9. HR risk scoring (Low / Medium / High) and export of high-risk employees

## Project Structure
```
.
├── attrition_prediction.py
├── WA_Fn-UseC_-HR-Employee-Attrition.csv
├── requirements.txt
└── README.md
```

## How to Run
```bash
git clone <your-repo-link>
cd <repo-folder>
pip install -r requirements.txt
python attrition_prediction.py
```

## Outputs Generated
| File | Description |
|------|-------------|
| `eda_overview.png`, `eda_categorical.png` | EDA charts |
| `model_comparison.csv` | Metrics for all models |
| `evaluation.png` | Confusion matrix and ROC curves |
| `feature_importance.png` | Top drivers of attrition |
| `high_risk_employees.csv` | Current employees flagged high-risk |
| `attrition_model.joblib` | Saved trained model and threshold |

## Key Insights
- Overtime, low monthly income, low stock options, young age and short tenure are strong attrition drivers.
- Employees flagged as high-risk can be targeted with retention actions: pay review, workload balancing, career growth plans.

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

## Author
Your Name
