import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ============================================================
# Load data (same as HW2)
# ============================================================
df = pd.read_csv("hw2_prepared_dataset.csv")

feature_cols = [
    "delivery_days", "delivery_vs_estimated", "price", "freight_value",
    "seller_state", "payment_type", "payment_installments", "payment_value",
    "product_weight_g", "product_length_cm", "product_height_cm",
    "product_width_cm", "customer_state", "total_cost", "log_price", "is_late"
]

X = df[feature_cols]
y = df["is_positive_review"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

num_cols = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
cat_cols = [c for c in X_train.columns if c not in num_cols]

preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols)
])

# ============================================================
# Create MLflow experiment
# ============================================================
mlflow.set_experiment("olist-satisfaction")

def log_run(model_name, pipeline, params):
    with mlflow.start_run(run_name=model_name):
        # Train
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        # Metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob)
        }

        # Log parameters, metrics, and model
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, "model")

        print(f"\n{model_name}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        return mlflow.active_run().info.run_id

# ============================================================
# Run 1: Logistic Regression (from HW2 baseline)
# ============================================================
lr_pipeline = Pipeline([
    ("preprocess", preprocess),
    ("model", LogisticRegression(max_iter=1000, random_state=42))
])

log_run("Logistic Regression", lr_pipeline, {
    "model_type": "LogisticRegression",
    "max_iter": 1000
})

# ============================================================
# Run 2: Tuned Random Forest (from HW2 best model)
# ============================================================
rf_pipeline = Pipeline([
    ("preprocess", preprocess),
    ("model", RandomForestClassifier(
        n_estimators=50,
        max_depth=15,
        min_samples_split=5,
        random_state=42
    ))
])

best_run_id = log_run("Tuned Random Forest", rf_pipeline, {
    "model_type": "RandomForestClassifier",
    "n_estimators": 50,
    "max_depth": 15,
    "min_samples_split": 5
})

# ============================================================
# Register best model
# ============================================================
model_uri = f"runs:/{best_run_id}/model"
result = mlflow.register_model(model_uri, "olist-satisfaction-model")
print(f"\nModel registered: {result.name}, version: {result.version}")

# Transition to Production stage
from mlflow.tracking import MlflowClient
client = MlflowClient()
client.transition_model_version_stage(
    name="olist-satisfaction-model",
    version=result.version,
    stage="Production"
)
print(f"Model version {result.version} transitioned to Production stage.")
print("\nDone! Now run: mlflow ui --port 5001")