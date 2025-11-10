import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from scipy.stats import randint, uniform
from sklearn.ensemble import StackingClassifier
import mlflow
import mlflow.sklearn
import time


def create_preprocessor():
    skewed_cols = ["ap_hi", "ap_lo", "BMI", "pulse_pressure", "height", "weight"]
    scaled_cols = ["age", "ap_hi", "ap_lo", "BMI", "pulse_pressure", "height", "weight"]
    preprocess = ColumnTransformer(
        transformers=[
            ("log", FunctionTransformer(np.log1p, validate=False), skewed_cols),
            ("scale", StandardScaler(), scaled_cols)
        ],
        remainder="passthrough"
    )
    return preprocess


def train_stacked_ensemble_pipeline(df):
    mlflow.set_experiment("Stacked_Ensemble")
    run_name = f"Stacked_Ensemble_Training_{time.strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name):
        X = df.drop(["cardio"], axis=1)
        y = df["cardio"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True
        )
        preprocess = create_preprocessor()
        xgb = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=42, n_jobs=-1, use_label_encoder=False
        )
        lgbm = LGBMClassifier(
            n_estimators=500, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=42, n_jobs=-1
        )
        cat = CatBoostClassifier(
            iterations=500, learning_rate=0.05, depth=6,
            l2_leaf_reg=3, subsample=0.8, eval_metric="AUC",
            random_seed=42, verbose=False
        )
        meta_model = XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=42, n_jobs=-1, use_label_encoder=False
        )
        stack = StackingClassifier(
            estimators=[
                ("xgb", xgb),
                ("lgbm", lgbm),
                ("cat", cat)
            ],
            final_estimator=meta_model,
            passthrough=True,
            n_jobs=-1
        )
        pipe = Pipeline([
            ("preprocess", preprocess),
            ("stack", stack)
        ])
        pipe.fit(X_train, y_train)
        y_train_pred = pipe.predict(X_train)
        y_test_pred = pipe.predict(X_test)
        metrics_df = pd.DataFrame({
            "Model": ["Stacked Ensemble", "Stacked Ensemble"],
            "Dataset": ["Train", "Test"],
            "Accuracy": [accuracy_score(y_train, y_train_pred), accuracy_score(y_test, y_test_pred)],
            "Precision": [precision_score(y_train, y_train_pred), precision_score(y_test, y_test_pred)],
            "Recall": [recall_score(y_train, y_train_pred), recall_score(y_test, y_test_pred)],
            "F1 Score": [f1_score(y_train, y_train_pred), f1_score(y_test, y_test_pred)],
            "ROC-AUC": [roc_auc_score(y_train, y_train_pred), roc_auc_score(y_test, y_test_pred)]
        })
        mlflow.log_metric("train_accuracy", accuracy_score(y_train, y_train_pred))
        mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_test_pred))
        mlflow.log_metric("train_precision", precision_score(y_train, y_train_pred))
        mlflow.log_metric("test_precision", precision_score(y_test, y_test_pred))
        mlflow.log_metric("train_recall", recall_score(y_train, y_train_pred))
        mlflow.log_metric("test_recall", recall_score(y_test, y_test_pred))
        mlflow.log_metric("train_f1", f1_score(y_train, y_train_pred))
        mlflow.log_metric("test_f1", f1_score(y_test, y_test_pred))
        mlflow.log_metric("train_roc_auc", roc_auc_score(y_train, y_train_pred))
        mlflow.log_metric("test_roc_auc", roc_auc_score(y_test, y_test_pred))
        mlflow.log_param("base_models", "['XGBoost', 'LightGBM', 'CatBoost']")
        mlflow.log_param("meta_model", "XGBoost")
        mlflow.log_param("stacking_method", "passthrough=True")
        print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
        print("Test Accuracy:", accuracy_score(y_test, y_test_pred), end="\n\n")
        print("Train Classification Report:\n\n", classification_report(y_train, y_train_pred))
        print("Test Classification Report:\n\n", classification_report(y_test, y_test_pred))
        print("Train AUC:", roc_auc_score(y_train, y_train_pred))
        print("Test AUC:", roc_auc_score(y_test, y_test_pred))
        cm = confusion_matrix(y_test, y_test_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["No Cardio (0)", "Cardio (1)"],
                    yticklabels=["No Cardio (0)", "Cardio (1)"])
        plt.title("Stacked Ensemble Confusion Matrix (Test Set)", fontsize=14)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()
        mlflow.sklearn.log_model(pipe, "stacked_ensemble_pipeline")
        return pipe, metrics_df


def train_catboost_pipeline(df):
    X = df.drop(["cardio"], axis=1)
    y = df["cardio"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True
    )
    preprocess = create_preprocessor()
    pipe = Pipeline([
        ("preprocess", preprocess),
        ("cat", CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3,
            subsample=0.8,
            eval_metric="AUC",
            random_seed=42,
            verbose=False
        ))
    ])
    pipe.fit(X_train, y_train)
    y_train_pred = pipe.predict(X_train)
    y_test_pred = pipe.predict(X_test)
    metrics_df = pd.DataFrame({
        "Model": ["CatBoost", "CatBoost"],
        "Dataset": ["Train", "Test"],
        "Accuracy": [accuracy_score(y_train, y_train_pred), accuracy_score(y_test, y_test_pred)],
        "Precision": [precision_score(y_train, y_train_pred), precision_score(y_test, y_test_pred)],
        "Recall": [recall_score(y_train, y_train_pred), recall_score(y_test, y_test_pred)],
        "F1 Score": [f1_score(y_train, y_train_pred), f1_score(y_test, y_test_pred)],
        "ROC-AUC": [roc_auc_score(y_train, y_train_pred), roc_auc_score(y_test, y_test_pred)]
    })
    print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
    print("Test Accuracy:", accuracy_score(y_test, y_test_pred), end="\n\n")
    print("Train Classification Report:\n\n", classification_report(y_train, y_train_pred))
    print("Test Classification Report:\n\n", classification_report(y_test, y_test_pred))
    print("Train AUC:", roc_auc_score(y_train, y_train_pred))
    print("Test AUC:", roc_auc_score(y_test, y_test_pred))
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["No Cardio (0)", "Cardio (1)"],
                yticklabels=["No Cardio (0)", "Cardio (1)"])
    plt.title("Cat Boost Confusion Matrix (Test Set)", fontsize=14)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
    return pipe, metrics_df


def save_model(pipe, path="models/catboost_pipeline.pkl"):
    joblib.dump(pipe, path)
    print(f"Pipeline saved at: {path}")


def load_model(path="models/catboost_pipeline.pkl"):
    return joblib.load(path)


def predict_with_pipeline(pipe, input_dict):
    if isinstance(input_dict, dict):
        input_df = pd.DataFrame([input_dict])
    else:
        input_df = input_dict.copy()
    prediction = pipe.predict(input_df)[0]
    probability = pipe.predict_proba(input_df)[0][1]
    mlflow.set_experiment("Cardio_Live_Predictions")
    with mlflow.start_run(run_name="Live_Prediction", nested=True):
        mlflow.log_param("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
        mlflow.log_param("input_data", str(input_df.to_dict(orient="records")[0]))
        mlflow.log_metric("predicted_label", int(prediction))
        mlflow.log_metric("predicted_probability", float(probability))
    return prediction, probability
