import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data():
    """Load wine quality dataset"""
    try:
        # You can download the dataset from:
        # Red wine: https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
        # White wine: https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv

        # For now, we'll create synthetic data for demonstration
        logger.info("Creating synthetic wine quality data...")

        # Create synthetic dataset similar to wine quality data
        np.random.seed(42)
        n_samples = 2000

        # Feature ranges based on typical wine quality dataset
        data = {
            'fixed_acidity': np.random.uniform(4.0, 16.0, n_samples),
            'volatile_acidity': np.random.uniform(0.1, 1.6, n_samples),
            'citric_acid': np.random.uniform(0.0, 1.0, n_samples),
            'residual_sugar': np.random.uniform(0.5, 15.0, n_samples),
            'chlorides': np.random.uniform(0.01, 0.2, n_samples),
            'free_sulfur_dioxide': np.random.uniform(1.0, 70.0, n_samples),
            'total_sulfur_dioxide': np.random.uniform(10.0, 300.0, n_samples),
            'density': np.random.uniform(0.99, 1.01, n_samples),
            'pH': np.random.uniform(2.8, 4.0, n_samples),
            'sulphates': np.random.uniform(0.3, 2.0, n_samples),
            'alcohol': np.random.uniform(8.0, 15.0, n_samples),
            'type': np.random.choice(['red', 'white'], n_samples)
        }

        # Create quality scores (3-9, but we'll classify into 4 categories)
        quality = np.random.choice([3, 4, 5, 6, 7, 8, 9], n_samples, p=[0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05])

        # Convert to DataFrame
        df = pd.DataFrame(data)
        df['quality'] = quality

        # Create quality classes (0-3)
        df['quality_class'] = pd.cut(df['quality'],
                                     bins=[0, 4, 5, 7, 10],
                                     labels=[0, 1, 2, 3])  # 0:Poor, 1:Average, 2:Good, 3:Excellent

        logger.info(f"Dataset created with {len(df)} samples")
        logger.info(f"Quality class distribution:\n{df['quality_class'].value_counts()}")

        return df

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def preprocess_data(df):
    """Preprocess the data for training"""
    # Create a copy
    data = df.copy()

    # Encode wine type
    le = LabelEncoder()
    data['type_encoded'] = le.fit_transform(data['type'])

    # Features to use
    features = [
        'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
        'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
        'pH', 'sulphates', 'alcohol', 'type_encoded'
    ]

    X = data[features]
    y = data['quality_class']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, le, features


def train_model(X_train, y_train):
    """Train the Random Forest model"""
    logger.info("Training Random Forest model...")

    # Define parameter grid for GridSearch
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Create Random Forest classifier
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')

    # Use GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    # Fit the model
    grid_search.fit(X_train, y_train)

    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def evaluate_model(model, X_test, y_test):
    """Evaluate the model performance"""
    logger.info("Evaluating model...")

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Test Accuracy: {accuracy:.4f}")

    # Classification report
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred,
                                      target_names=['Poor', 'Average', 'Good', 'Excellent']))

    # Confusion matrix
    logger.info("\nConfusion Matrix:")
    logger.info(confusion_matrix(y_test, y_pred))

    return accuracy


def save_model(model, scaler, label_encoder, features):
    """Save the trained model and preprocessing objects"""
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Save the model
    model_path = models_dir / "best_model.pkl"
    joblib.dump(model, model_path)

    # Save preprocessing objects
    preprocessing_path = models_dir / "preprocessing.pkl"
    joblib.dump({
        'scaler': scaler,
        'label_encoder': label_encoder,
        'features': features
    }, preprocessing_path)

    logger.info(f"Model saved to {model_path}")
    logger.info(f"Preprocessing objects saved to {preprocessing_path}")


def main():
    """Main function to train and save the model"""
    try:
        # Load data
        df = load_data()

        # Preprocess data
        X_train, X_test, y_train, y_test, scaler, le, features = preprocess_data(df)

        # Train model
        model = train_model(X_train, y_train)

        # Evaluate model
        evaluate_model(model, X_test, y_test)

        # Save model
        save_model(model, scaler, le, features)

        logger.info("Model training completed successfully!")

    except Exception as e:
        logger.error(f"Error in model training: {e}")
        raise


if __name__ == "__main__":
    main()