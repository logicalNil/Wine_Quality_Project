import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(red_path, white_path):
    """Load and combine red and white wine datasets"""
    # Check if files exist
    if not os.path.exists(red_path):
        raise FileNotFoundError(f"Red wine data file not found: {red_path}")
    if not os.path.exists(white_path):
        raise FileNotFoundError(f"White wine data file not found: {white_path}")

    try:
        red = pd.read_csv(red_path, sep=';')
        white = pd.read_csv(white_path, sep=';')

        logger.info(f"Red wine data shape: {red.shape}")
        logger.info(f"White wine data shape: {white.shape}")

        # Add wine type column
        red['type'] = 'red'
        white['type'] = 'white'

        # Combine datasets
        df = pd.concat([red, white], ignore_index=True)

        # Convert type to categorical
        df['type'] = df['type'].map({'red': 0, 'white': 1})

        return df

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def preprocess_data(df):
    """Preprocess the wine dataset"""
    # Handle missing values if any
    df = df.dropna()

    # Create quality categories (0-2: poor, 3-4: average, 5-6: good, 7-10: excellent)
    df['quality_category'] = pd.cut(
        df['quality'],
        bins=[0, 2, 4, 6, 10],
        labels=[0, 1, 2, 3]
    )

    # Drop the original quality column
    df = df.drop('quality', axis=1)

    return df


def prepare_datasets():
    """Prepare and split the dataset"""
    # Define paths - using absolute paths from project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    red_path = os.path.join(project_root, 'data', 'winequality-red.csv')
    white_path = os.path.join(project_root, 'data', 'winequality-white.csv')

    logger.info(f"Looking for red wine data at: {red_path}")
    logger.info(f"Looking for white wine data at: {white_path}")

    # Load data
    df = load_data(red_path, white_path)

    # Preprocess data
    df = preprocess_data(df)

    # Split features and target
    X = df.drop('quality_category', axis=1)
    y = df['quality_category']

    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create data directory if it doesn't exist
    data_dir = os.path.join(project_root, 'data')
    os.makedirs(data_dir, exist_ok=True)

    # Save processed data
    X_train.to_csv(os.path.join(data_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(data_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(data_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(data_dir, 'y_test.csv'), index=False)

    logger.info(f"Training set size: {X_train.shape}")
    logger.info(f"Test set size: {X_test.shape}")
    logger.info(f"Class distribution: {np.bincount(y_train)}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    prepare_datasets()