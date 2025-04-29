# Indian Railway Accidents Analysis & Prediction

![Railway Safety](https://img.shields.io/badge/Railway-Safety-red) ![Data Mining](https://img.shields.io/badge/Data-Mining-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-App-orange) ![Python](https://img.shields.io/badge/Python-3.11-green)

A comprehensive data mining application for analyzing railway accidents and predicting their severity using historical Indian railway accident data from 1902 to 2024.

## Project Overview

This application helps analyze patterns in railway accidents across India, identify high-risk locations, predict accident severity, detect anomalies, and understand temporal trends. The interactive dashboard provides valuable insights for railway safety planning and risk mitigation.

## Features

- **Data Overview**: Visualize accident statistics, data distributions, and quality metrics
- **Severity Prediction**: ML-based prediction of accident severity (Low/Medium/High) 
- **Geospatial Analysis**: Interactive map of accident hotspots with DBSCAN clustering
- **Temporal Trends**: Time series analysis with trend-seasonal decomposition
- **Anomaly Detection**: Identification of unusual accident patterns

## Project Structure

```
├── app.py                  # Main Streamlit application
├── data/                   # Dataset directory
│   └── indian_railway_accidents.csv  # Railway accident records
├── models/                 # ML model implementations
│   ├── anomaly_detection.py  # Anomaly detection model
│   └── severity_model.py     # Accident severity prediction
├── utils/                  # Utility modules
│   ├── data_preprocessing.py  # Data cleaning and preparation
│   ├── geocoding.py           # Location geocoding functionality
│   ├── modeling.py            # Model training utilities
│   └── visualization.py       # Data visualization functions
└── README.md               # Project documentation
```

## Technologies Used

- **Python 3.11**: Core programming language
- **Streamlit**: Interactive web application framework
- **Pandas & NumPy**: Data manipulation and numerical processing
- **Scikit-learn**: Machine learning models and clustering algorithms
- **Plotly**: Interactive data visualizations
- **Statsmodels**: Time series decomposition and analysis
- **GeoPy**: Geocoding for location data
- **XGBoost**: Advanced gradient boosting for severity prediction

## Installation & Usage

1. Clone the repository
```bash
git clone https://github.com/yourusername/RailwayAccidentAnalyzer.git
cd indian-railway-accidents-analysis
```

2. Install required packages
```bash
pip install streamlit pandas numpy scikit-learn plotly statsmodels geopy xgboost joblib mlxtend
```

3. Run the application
```bash
streamlit run app.py
```

4. Navigate to http://localhost:5000 in your browser

## Key Analysis Methods

### 1. Severity Prediction
Random Forest classification model predicts accident severity based on accident type, cause, location, and time period, categorized as:
- **Low**: ≤10 fatalities
- **Medium**: 10-50 fatalities  
- **High**: >50 fatalities

### 2. Geospatial Hotspot Analysis
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm identifies geographical clusters of accidents, with customizable radius and minimum accidents parameters.

### 3. Temporal Trend Analysis
Time series analysis with STL (Seasonal-Trend decomposition using LOESS) breaks down accident patterns into:
- Long-term trends
- Seasonal components
- Residual variations

### 4. Anomaly Detection
Isolation Forest algorithm identifies outlier accidents by:
- Analyzing relationships between multiple variables
- Detecting unusual combinations of accident properties
- Highlighting accidents with anomalous statistics

## Dataset

The dataset contains historical Indian railway accidents spanning over a century (1902-2024) with the following key attributes:
- Date and location information
- Accident type (derailment, collision, fire, etc.)
- Cause of accident
- Fatalities and injuries
- Train details

## Limitations

- Historical data may have gaps, especially for older records
- Geocoding accuracy depends on location name standardization
- Prediction models are based on historical patterns and may require updates as new data becomes available

## Future Enhancements

Potential areas for expansion:
- Integration with real-time railway data
- Additional predictive models for accident risk assessment
- Natural language processing for accident description analysis
- Mobile-friendly interface for field use

## License

This project is available under the MIT License.
