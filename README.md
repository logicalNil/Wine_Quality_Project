# Wine Quality MLOps Project

## 📋 Project Overview

This project implements a complete MLOps workflow for multiclass classification of wine quality based on physicochemical properties. The system predicts whether a wine is "poor", "average", or "excellent" quality based on its characteristics.

### Key Features

- Data versioning with DVC
- Experiment tracking with MLflow
- REST API with FastAPI
- Containerization with Docker
- CI/CD with GitHub Actions
- Monitoring with Prometheus and Grafana
- User-friendly Streamlit frontend

## 🏗️ Architecture

```
Data Sources → Data Preparation → Model Training → API Deployment → Frontend → Monitoring
    │              │                 │               │               │           │
    │           (DVC)           (MLflow)        (Docker)       (Streamlit)  (Prometheus/
    │                                                              │         Grafana)
GitHub Actions CI/CD Pipeline
```

## 📁 Project Structure

```
wine-quality-mlops/
├── data/                 # Raw and processed datasets
├── src/                  # Data preparation and training code
│   ├── data_prep.py      # Data loading and preprocessing
│   ├── train.py          # Model training with MLflow
│   └── evaluate.py       # Model evaluation
├── api/                  # FastAPI application
│   ├── main.py           # API endpoints
│   └── schemas.py        # Pydantic models
├── tests/                # Test files
├── models/               # Trained models
├── logs/                 # Application logs
├── .github/
│   └── workflows/
│       └── ci.yml        # GitHub Actions workflow
├── Dockerfile           # Docker configuration
├── docker-compose.yml   # Multi-container setup
├── requirements.txt     # Python dependencies
├── app.py              # Streamlit frontend
└── README.md           # Project documentation
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Docker and Docker Compose
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/wine-quality-mlops.git
   cd wine-quality-mlops
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the datasets**
   - Download `winequality-red.csv` and `winequality-white.csv` from the UCI Machine Learning Repository
   - Place them in the `data/` directory

4. **Prepare the data**
   ```bash
   python src/data_prep.py
   ```

5. **Train the models**
   ```bash
   python src/train.py
   ```

6. **Start the services with Docker Compose**
   ```bash
   docker-compose up -d
   ```

7. **Run the frontend application**
   ```bash
   streamlit run app.py
   ```

## 🎯 Usage

1. Open the frontend application in your browser (usually http://localhost:8501)
2. Adjust the sliders to input wine properties or use the sample data provided
3. Click "Predict Quality" to get a prediction
4. View the predicted quality class and probability distribution

## 🔌 API Endpoints

The FastAPI service provides the following endpoints:

- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Predict wine quality
- `GET /metrics` - Prometheus metrics

### Example API Request

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
  "fixed_acidity": 7.4,
  "volatile_acidity": 0.7,
  "citric_acid": 0.0,
  "residual_sugar": 1.9,
  "chlorides": 0.076,
  "free_sulfur_dioxide": 11.0,
  "total_sulfur_dioxide": 34.0,
  "density": 0.9978,
  "pH": 3.51,
  "sulphates": 0.56,
  "alcohol": 9.4,
  "type": "red"
}'
```

### Example API Response

```json
{
  "predicted_class": "average",
  "confidence": 0.85,
  "probabilities": {
    "poor": 0.10,
    "average": 0.85,
    "excellent": 0.05
  },
  "latency": 0.045
}
```

## 📊 Monitoring

The project includes comprehensive monitoring:

### MLflow Tracking: http://localhost:5000
- Experiment comparison
- Parameter and metric tracking
- Model versioning

### Prometheus: http://localhost:9090
- API metrics collection
- Performance monitoring

### Grafana: http://localhost:3000 (admin/admin)
- Visualization dashboards
- Real-time monitoring

## 🔧 CI/CD Pipeline

The GitHub Actions workflow includes:

- Code linting with flake8
- Automated testing with pytest
- Docker image building
- Container pushing to Docker Hub
- Automated deployment (if configured)

## 🧪 Testing

Run the test suite with:

```bash
python -m pytest tests/ -v
```

## 📈 Model Performance

The project compares multiple machine learning algorithms:

- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

### Performance metrics tracked:
- Accuracy
- Precision
- Recall
- F1 Score
- Feature importance

## 🛠️ Technologies Used

- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Experiment Tracking**: MLflow
- **API Framework**: FastAPI, Pydantic
- **Containerization**: Docker
- **CI/CD**: GitHub Actions
- **Monitoring**: Prometheus, Grafana
- **Frontend**: Streamlit
- **Data Versioning**: DVC (optional)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the Wine Quality dataset
- The MLOps community for best practices and patterns

## 📞 Support

If you have any questions or issues, please open an issue on the GitHub repository.

<img src="images/Output Frontend.png" alt="Description" width="600" height="400">
