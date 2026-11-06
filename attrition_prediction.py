"""
Employee Attrition Prediction to Support HR Decision-Making
Dataset: IBM HR Analytics Employee Attrition & Performance (Kaggle)
File   : WA_Fn-UseC_-HR-Employee-Attrition.csv  (place next to this script)

Install: pip install pandas numpy scikit-learn matplotlib seaborn joblib
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report,
                             roc_curve, precision_recall_curve)

RANDOM_STATE = 42
DATA_PATH = "WA_Fn-UseC_-HR-Employee-Attrition.csv"

# ------------------------------------------------------------------
# 1. LOAD & INSPECT
# ------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())
print(df["Attrition"].value_counts(normalize=True).round(3))

# ------------------------------------------------------------------
# 2. CLEANING
# ------------------------------------------------------------------
# Constant / ID columns carry no information
drop_cols = [c for c in ["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"] if c in df.columns]
df = df.drop(columns=drop_cols)
df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

# ------------------------------------------------------------------
# 3. EDA (saved as images for the report)
# ------------------------------------------------------------------
sns.set_style("whitegrid")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
df["Attrition"].value_counts().plot.bar(ax=ax[0], color=["#4c72b0", "#c44e52"])
ax[0].set_title("Attrition Class Distribution"); ax[0].set_xticklabels(["Stayed", "Left"], rotation=0)
sns.heatmap(df.select_dtypes("number").corr()[["Attrition"]].sort_values("Attrition", ascending=False),
            annot=True, fmt=".2f", cmap="coolwarm", ax=ax[1])
ax[1].set_title("Correlation with Attrition")
plt.tight_layout(); plt.savefig("eda_overview.png", dpi=150); plt.close()

cat_eda = ["OverTime", "JobRole", "MaritalStatus", "BusinessTravel", "Department"]
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for a, col in zip(axes.ravel(), cat_eda):
    rate = df.groupby(col)["Attrition"].mean().sort_values()
    rate.plot.barh(ax=a, color="#c44e52"); a.set_title(f"Attrition rate by {col}")
axes.ravel()[-1].axis("off")
plt.tight_layout(); plt.savefig("eda_categorical.png", dpi=150); plt.close()

# ------------------------------------------------------------------
# 4. FEATURE ENGINEERING
# ------------------------------------------------------------------
df["IncomePerYearExp"] = df["MonthlyIncome"] / (df["TotalWorkingYears"] + 1)
df["YearsSincePromoRatio"] = df["YearsSinceLastPromotion"] / (df["YearsAtCompany"] + 1)
df["AvgTenurePerCompany"] = df["TotalWorkingYears"] / (df["NumCompaniesWorked"] + 1)
df["SatisfactionScore"] = df[["JobSatisfaction", "EnvironmentSatisfaction",
                              "RelationshipSatisfaction", "WorkLifeBalance"]].mean(axis=1)

# ------------------------------------------------------------------
# 5. PREPROCESSING
# ------------------------------------------------------------------
X = df.drop(columns="Attrition")
y = df["Attrition"]
cat_cols = X.select_dtypes("object").columns.tolist()
num_cols = X.select_dtypes("number").columns.tolist()

preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

# ------------------------------------------------------------------
# 6. MODEL COMPARISON (class_weight handles the ~16% imbalance)
# ------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                            random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
rows, fitted = [], {}
for name, clf in models.items():
    pipe = Pipeline([("prep", preprocess), ("clf", clf)])
    cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc").mean()
    pipe.fit(X_train, y_train)
    fitted[name] = pipe
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    rows.append({"Model": name, "CV AUC": cv_auc,
                 "Accuracy": accuracy_score(y_test, pred),
                 "Precision": precision_score(y_test, pred),
                 "Recall": recall_score(y_test, pred),
                 "F1": f1_score(y_test, pred),
                 "Test AUC": roc_auc_score(y_test, proba)})

results = pd.DataFrame(rows).set_index("Model").round(3)
print("\n=== Model comparison ===\n", results)
results.to_csv("model_comparison.csv")

# ------------------------------------------------------------------
# 7. TUNE THE BEST CANDIDATE (Logistic Regression is usually strong here)
# ------------------------------------------------------------------
best_name = results["Test AUC"].idxmax()
print("\nBest model by Test AUC:", best_name)

grids = {
    "Logistic Regression": {"clf__C": [0.01, 0.05, 0.1, 0.5, 1, 5]},
    "Random Forest": {"clf__max_depth": [5, 8, None], "clf__min_samples_leaf": [1, 3, 5]},
    "Gradient Boosting": {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1],
                          "clf__max_depth": [2, 3]},
}
gs = GridSearchCV(Pipeline([("prep", preprocess), ("clf", models[best_name])]),
                  grids[best_name], cv=cv, scoring="roc_auc", n_jobs=-1)
gs.fit(X_train, y_train)
final = gs.best_estimator_
print("Best params:", gs.best_params_)

# ------------------------------------------------------------------
# 8. THRESHOLD TUNING - HR cares more about catching leavers (recall)
# ------------------------------------------------------------------
proba = final.predict_proba(X_test)[:, 1]
prec, rec, thr = precision_recall_curve(y_test, proba)
f2 = (5 * prec * rec) / (4 * prec + rec + 1e-9)          # F2 weights recall higher
best_thr = thr[np.argmax(f2[:-1])]
pred = (proba >= best_thr).astype(int)
print(f"\nChosen threshold (max F2): {best_thr:.2f}")
print(classification_report(y_test, pred, target_names=["Stayed", "Left"]))

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt="d", cmap="Blues", ax=ax[0],
            xticklabels=["Stayed", "Left"], yticklabels=["Stayed", "Left"])
ax[0].set_title("Confusion Matrix"); ax[0].set_xlabel("Predicted"); ax[0].set_ylabel("Actual")
for name, pipe in fitted.items():
    fpr, tpr, _ = roc_curve(y_test, pipe.predict_proba(X_test)[:, 1])
    ax[1].plot(fpr, tpr, label=f"{name} ({roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1]):.2f})")
ax[1].plot([0, 1], [0, 1], "k--"); ax[1].set_title("ROC Curves"); ax[1].legend()
plt.tight_layout(); plt.savefig("evaluation.png", dpi=150); plt.close()

# ------------------------------------------------------------------
# 9. EXPLAINABILITY - what drives attrition?
# ------------------------------------------------------------------
perm = permutation_importance(final, X_test, y_test, scoring="roc_auc",
                              n_repeats=15, random_state=RANDOM_STATE, n_jobs=-1)
imp = pd.Series(perm.importances_mean, index=X_test.columns).sort_values(ascending=False).head(12)
imp.sort_values().plot.barh(figsize=(8, 5), color="#4c72b0")
plt.title("Top Drivers of Attrition (Permutation Importance)")
plt.tight_layout(); plt.savefig("feature_importance.png", dpi=150); plt.close()
print("\nTop drivers:\n", imp.round(4))

# ------------------------------------------------------------------
# 10. HR DECISION SUPPORT - risk scoring of current employees
# ------------------------------------------------------------------
df_scored = df.copy()
df_scored["AttritionRisk"] = final.predict_proba(X)[:, 1]
df_scored["RiskBand"] = pd.cut(df_scored["AttritionRisk"], [0, 0.3, 0.6, 1.0],
                               labels=["Low", "Medium", "High"], include_lowest=True)
print("\nRisk band distribution:\n", df_scored["RiskBand"].value_counts())
print("\nRisk by department:\n", df_scored.groupby("Department")["AttritionRisk"].mean().round(3))

high_risk = df_scored[(df_scored["RiskBand"] == "High") & (df_scored["Attrition"] == 0)]
high_risk.sort_values("AttritionRisk", ascending=False).to_csv("high_risk_employees.csv", index=False)
print(f"\n{len(high_risk)} current employees flagged high-risk -> high_risk_employees.csv")

# ------------------------------------------------------------------
# 11. SAVE MODEL
# ------------------------------------------------------------------
joblib.dump({"model": final, "threshold": best_thr}, "attrition_model.joblib")
print("Saved attrition_model.joblib")
