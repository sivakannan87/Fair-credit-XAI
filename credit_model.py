"""
Credit Risk Model Training & Evaluation Engine.
Builds a high-accuracy, interpretable ensemble classifier (Random Forest / Gradient Boosting)
for loan approval probability estimation.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from data_generator import generate_credit_data

FEATURES_NUM = [
    "cibil_score",
    "monthly_income_inr",
    "debt_to_income_ratio",
    "loan_amount_inr",
    "tenure_months",
    "existing_loans_count",
    "digital_banking_score",
    "age"
]

CATEGORICAL_COLS = ["employment_type"]
SENSITIVE_COLS = ["gender", "demographic_region"]

def prepare_data(df: pd.DataFrame):
    # One-hot encode categorical features without dropping sensitive cols for auditing
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)
    
    feature_cols = [c for c in df_encoded.columns if c in FEATURES_NUM or c.startswith("employment_type_")]
    X = df_encoded[feature_cols]
    y = df_encoded["approved"]
    
    return X, y, df_encoded, feature_cols

def train_credit_model(data_path: str = "loan_data.csv", model_dir: str = "models"):
    os.makedirs(model_dir, exist_ok=True)
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        df = generate_credit_data()
        df.to_csv(data_path, index=False)

    X, y, df_encoded, feature_cols = prepare_data(df)
    
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df_encoded, test_size=0.25, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.08,
        max_depth=4,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "test_samples": len(y_test)
    }

    # Save artifacts
    joblib.dump(model, os.path.join(model_dir, "credit_model.pkl"))
    joblib.dump(feature_cols, os.path.join(model_dir, "feature_cols.pkl"))
    joblib.dump(metrics, os.path.join(model_dir, "metrics.pkl"))
    df_test.to_csv(os.path.join(model_dir, "test_dataset.csv"), index=False)

    print("Model Training Completed.")
    print(f"Accuracy: {metrics['accuracy']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f} | F1: {metrics['f1']:.4f}")
    return model, feature_cols, metrics

if __name__ == "__main__":
    train_credit_model()
