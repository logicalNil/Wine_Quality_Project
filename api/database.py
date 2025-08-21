import sqlite3
import json
from datetime import datetime


def init_db():
    """Initialize the SQLite database"""
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()

    # Create predictions table
    c.execute('''
              CREATE TABLE IF NOT EXISTS predictions
              (
                  id
                  INTEGER
                  PRIMARY
                  KEY
                  AUTOINCREMENT,
                  timestamp
                  DATETIME
                  DEFAULT
                  CURRENT_TIMESTAMP,
                  fixed_acidity
                  REAL,
                  volatile_acidity
                  REAL,
                  citric_acid
                  REAL,
                  residual_sugar
                  REAL,
                  chlorides
                  REAL,
                  free_sulfur_dioxide
                  REAL,
                  total_sulfur_dioxide
                  REAL,
                  density
                  REAL,
                  pH
                  REAL,
                  sulphates
                  REAL,
                  alcohol
                  REAL,
                  type
                  INTEGER,
                  predicted_class
                  INTEGER,
                  probabilities
                  TEXT,
                  latency
                  REAL,
                  client_ip
                  TEXT
              )
              ''')

    conn.commit()
    conn.close()


def log_prediction(features, predicted_class, probabilities, latency, client_ip):
    """Log prediction to SQLite database"""
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()

    c.execute('''
              INSERT INTO predictions
              (fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
               free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol, type,
               predicted_class, probabilities, latency, client_ip)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
              ''', (
                  features['fixed_acidity'],
                  features['volatile_acidity'],
                  features['citric_acid'],
                  features['residual_sugar'],
                  features['chlorides'],
                  features['free_sulfur_dioxide'],
                  features['total_sulfur_dioxide'],
                  features['density'],
                  features['pH'],
                  features['sulphates'],
                  features['alcohol'],
                  features['type'],
                  predicted_class,
                  json.dumps(probabilities),
                  latency,
                  client_ip
              ))

    conn.commit()
    conn.close()


# Initialize database on import
init_db()