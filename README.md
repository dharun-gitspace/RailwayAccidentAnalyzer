# Indian Railway Accidents Analysis & Prediction

![Railway Accidents Analysis](https://img.shields.io/badge/Railway-Safety-red) ![Data Mining](https://img.shields.io/badge/Data-Mining-blue) ![Python 3.11](https://img.shields.io/badge/Python-3.11-green) ![Streamlit](https://img.shields.io/badge/Streamlit-App-orange)

A comprehensive data mining application for analyzing and predicting Indian railway accidents using historical data from 1902-2024.

## 📋 Features

- **Data Overview**: Explore accident statistics, distributions, and data quality metrics
- **Severity Prediction**: Predict accident severity based on type, cause, location, and time period
- **Geospatial Analysis**: Visualize accident hotspots across India with DBSCAN clustering
- **Temporal Trends**: Analyze patterns over time with time series decomposition
- **Anomaly Detection**: Identify unusual accidents that deviate from typical patterns

## 📊 Screenshots

*Geospatial Hotspot Analysis*
![Geospatial Analysis](https://i.imgur.com/placeholder-image.png)

*Temporal Trend Analysis*
![Temporal Trends](https://i.imgur.com/placeholder-image.png)

## 🔧 Technologies Used

- **Python 3.11** - Core programming language
- **Streamlit** - Web application framework
- **Pandas & NumPy** - Data processing
- **Scikit-learn** - Machine learning models and clustering
- **XGBoost** - Gradient boosting for severity prediction
- **Plotly** - Interactive data visualization
- **Statsmodels** - Time series analysis
- **GeoPy** - Geocoding for location data

## 🏗️ Project Structure

```
├── app.py                  # Main Streamlit application
├── data/                   # Data directory
│   └── indian_railway_accidents.csv   # Historical accident data
├── models/                 # Model implementations
│   ├── anomaly_detection.py  # Anomaly detection model
│   └── severity_model.py     # Severity prediction model
├── utils/                  # Utility modules
│   ├── data_preprocessing.py  # Data cleaning and preparation
│   ├── geocoding.py           # Location geocoding utilities
│   ├── modeling.py            # Model training functions
│   └── visualization.py       # Data visualization functions
└── README.md               # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- Required Python packages (installable via pip):
  - streamlit
  - pandas
  - numpy
  - scikit-learn
  - plotly
  - xgboost
  - statsmodels
  - geopy
  - joblib

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/indian-railway-accidents-analysis.git
cd indian-railway-accidents-analysis
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the application
```bash
streamlit run app.py
```

4. Open your browser and go to `http://localhost:5000`

## 📈 Analysis Methods

### Severity Prediction
The system uses a Random Forest classifier to predict accident severity (Low, Medium, High) based on multiple factors including accident type, cause, location, and time period.

### Geospatial Hotspot Analysis
DBSCAN clustering identifies accident hotspots across India, allowing for targeted safety improvements in high-risk areas.

### Temporal Trend Analysis
Time series decomposition separates long-term trends from seasonal patterns and residuals, revealing how accident patterns have evolved over time.

### Anomaly Detection
Isolation Forest algorithm identifies unusual accidents that deviate from typical patterns, which may represent reporting errors, extreme events, or special circumstances.

## 📚 Dataset

The dataset contains historical Indian railway accidents from 1902 to 2024, including:

- Accident date and location
- Type of accident (derailment, collision, etc.)
- Cause of accident
- Number of fatalities and injuries
- Trains involved

## 🔒 Privacy & Ethics

This project uses historical data for analytical purposes only. All data is anonymized and focused on accident statistics rather than personal information.

## 👥 Contributing

Contributions to improve the analysis or extend the project are welcome. Please feel free to submit a pull request or open an issue to discuss potential changes.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- Indian Railways for safety initiatives
- Open data community for access to historical records
- Data mining and railway safety researchers