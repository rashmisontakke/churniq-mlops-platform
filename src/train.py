# src/train.py
# WHAT CHANGED:
# - LogisticRegression → XGBoost (much better on tabular data)
# - pd.get_dummies → sklearn Pipeline (prevents data leakage)
# - Added feature_engineering.py integration (fixes Bug 3)
# - Added SMOTE for class imbalance
# - Added SHAP TreeExplainer (fixes Bug 1 — now using tree model)
# - Saves pipeline.pkl instead of model.pkl + model_columns.pkl

import logging
import warnings
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, classification_report,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb

from feature_engineering import engineer_features

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "Customer-Churn-Records.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


def load_data() -> tuple:
    """Load data, apply feature engineering, return X and y."""
    logger.info(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna()

    # Apply your custom feature engineering
    df = engineer_features(df)
    logger.info(f"Dataset shape after feature engineering: {df.shape}")
    logger.info(f"Churn rate: {df['Exited'].mean():.2%}")

    feature_cols = [
        "Geography", "Gender", "CreditScore", "Age", "Tenure",
        "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
        "EstimatedSalary",
        # Your 5 engineered features:
        "balance_salary_ratio", "products_per_tenure",
        "age_bucket", "is_zero_balance", "engagement_score",
    ]

    X = df[feature_cols]
    y = df["Exited"].astype(int)
    return X, y, feature_cols


def build_pipeline() -> ImbPipeline:
    """
    sklearn Pipeline: preprocessing → SMOTE → XGBoost.
    ImbPipeline ensures SMOTE only runs on training data, not validation.
    """
    categorical_features = ["Geography", "Gender"]
    numerical_features = [
        "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
        "HasCrCard", "IsActiveMember", "EstimatedSalary",
        "balance_salary_ratio", "products_per_tenure",
        "age_bucket", "is_zero_balance", "engagement_score",
    ]

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
    ])

    return ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42, sampling_strategy=0.5)),
        ("classifier", xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def train():
    """Full training run with MLflow logging."""
    mlflow.set_experiment("churn-xgboost")

    X, y, feature_cols = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run() as run:
        logger.info("Training XGBoost pipeline...")
        pipeline = build_pipeline()
        pipeline.fit(X_train, y_train)

        # Evaluate
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1_score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
            "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
        }
        mlflow.log_params({"model": "XGBoost", "n_estimators": 200, "max_depth": 6})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, "model",
                                 registered_model_name="ChurnXGBoost")

        print("\n" + classification_report(y_test, y_pred,
                                           target_names=["Stay", "Churn"]))

        # Build SHAP explainer (TreeExplainer works with XGBoost)
        logger.info("Building SHAP explainer...")
        preprocessor = pipeline.named_steps["preprocessor"]
        classifier = pipeline.named_steps["classifier"]
        X_test_transformed = preprocessor.transform(X_test)

        explainer = shap.TreeExplainer(classifier)

        # Get feature names after one-hot encoding
        num_names = [
            "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
            "HasCrCard", "IsActiveMember", "EstimatedSalary",
            "balance_salary_ratio", "products_per_tenure",
            "age_bucket", "is_zero_balance", "engagement_score",
        ]
        cat_names = list(
            preprocessor.named_transformers_["cat"]
            .get_feature_names_out(["Geography", "Gender"])
        )
        all_feature_names = num_names + cat_names

        # Save everything
        joblib.dump(pipeline, MODELS_DIR / "pipeline.pkl")
        joblib.dump(explainer, MODELS_DIR / "explainer.pkl")
        joblib.dump({
            "feature_cols": feature_cols,
            "all_feature_names": all_feature_names,
        }, MODELS_DIR / "feature_names.pkl")

        import json
        with open(MODELS_DIR / "latest_metrics.json", "w") as f:
            json.dump({**metrics, "run_id": run.info.run_id}, f, indent=2)

        logger.info(f"F1: {metrics['f1_score']} | ROC-AUC: {metrics['roc_auc']}")
        logger.info(f"Pipeline saved to models/pipeline.pkl")
        return metrics


if __name__ == "__main__":
    results = train()
    print(f"\nF1 Score : {results['f1_score']}")
    print(f"ROC-AUC  : {results['roc_auc']}")