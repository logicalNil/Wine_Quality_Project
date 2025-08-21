import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("wine-quality-classification")


def load_data():
    """Load processed data"""
    X_train = pd.read_csv('../data/X_train.csv')
    X_test = pd.read_csv('../data/X_test.csv')
    y_train = pd.read_csv('../data/y_train.csv').squeeze()
    y_test = pd.read_csv('../data/y_test.csv').squeeze()

    return X_train, X_test, y_train, y_test


def train_model(model, model_name, params, X_train, y_train, X_test, y_test):
    """Train a model and log metrics with MLflow"""
    with mlflow.start_run(run_name=model_name):
        # Log parameters
        mlflow.log_params(params)

        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Log metrics
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Calculate feature importance for tree-based models
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            # Log feature importance as artifact
            feature_importance.to_csv('feature_importance.csv', index=False)
            mlflow.log_artifact('feature_importance.csv')

        logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

        return model, accuracy, f1


def main():
    """Main training function"""
    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Define models and parameters
    models = {
        "LogisticRegression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "params": {"max_iter": 1000, "random_state": 42}
        },
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {"n_estimators": 100, "random_state": 42}
        },
        "GradientBoosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42}
        }
    }

    best_model = None
    best_score = 0

    # Train and evaluate each model
    for name, config in models.items():
        model, accuracy, f1 = train_model(
            config["model"], name, config["params"],
            X_train, y_train, X_test, y_test
        )

        # Track best model
        if f1 > best_score:
            best_score = f1
            best_model = model

    # Save the best model
    joblib.dump(best_model, '../models/best_model.pkl')
    logger.info(f"Best model saved with F1 score: {best_score:.4f}")


if __name__ == "__main__":
    main()