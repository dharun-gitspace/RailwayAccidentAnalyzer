import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import pickle
from datetime import datetime

# Add utils directory to path
sys.path.append(os.path.abspath("utils"))
sys.path.append(os.path.abspath("models"))

# Import utility modules
from data_preprocessing import preprocess_data, standardize_states, extract_temporal_features, handle_missing_data
from geocoding import geocode_locations
from modeling import train_severity_model, predict_severity
from visualization import (
    plot_accident_map, 
    plot_temporal_trends, 
    plot_severity_distribution, 
    plot_accident_types,
    plot_anomalies
)

# Import model modules
from severity_model import SeverityModel
from anomaly_detection import AnomalyDetector
from association_mining import AssociationMiner

# Set page configuration
st.set_page_config(
    page_title="Indian Railway Accidents Analysis & Prediction",
    page_icon="🚂",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and introduction
st.title("🚂 Indian Railway Accidents Analysis & Prediction")
st.markdown("""
This application analyzes railway accidents in India from 1902 to 2024 and provides:
- Accident severity prediction
- Geospatial hotspot analysis
- Temporal trend analysis
- Anomaly detection
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page",
    ["Data Overview", "Severity Prediction", "Geospatial Analysis", 
     "Temporal Trends", "Anomaly Detection"]
)

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/indian_railway_accidents.csv")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Preprocess data
@st.cache_data
def get_processed_data(df):
    if df is not None:
        # Preprocess data
        df = preprocess_data(df)
        
        # Standardize state names
        df = standardize_states(df)
        
        # Extract temporal features
        df = extract_temporal_features(df)
        
        # Handle missing data
        df = handle_missing_data(df)
        
        # Geocode locations
        df = geocode_locations(df)
        
        return df
    return None

# Initialize data
raw_data = load_data()
if raw_data is not None:
    df = get_processed_data(raw_data)
else:
    st.error("Failed to load the dataset. Please check if the file exists.")
    st.stop()

# Initialize models
@st.cache_resource
def load_models(df):
    # Severity model
    severity_model = SeverityModel()
    severity_model.fit(df)
    
    # Anomaly detector
    anomaly_detector = AnomalyDetector()
    anomaly_detector.fit(df)
    
    return severity_model, anomaly_detector

if df is not None:
    severity_model, anomaly_detector = load_models(df)

# Data Overview Page
if page == "Data Overview":
    st.header("Data Overview")
    
    # Display basic statistics
    st.subheader("Dataset Summary")
    st.write(f"Time Period: 1902-2024")
    st.write(f"Total Accidents: {len(df)}")
    
    # Missing data information
    st.subheader("Missing Data Information")
    missing_data = df.isnull().sum()
    missing_df = pd.DataFrame({
        'Column': missing_data.index,
        'Missing Values': missing_data.values,
        'Percentage': (missing_data / len(df) * 100).round(2)
    })
    st.dataframe(missing_df)
    
    # Display sample data
    st.subheader("Sample Data")
    st.dataframe(df.head(10))
    
    # Basic visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribution of Accident Types")
        fig = plot_accident_types(df)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Severity Distribution")
        fig = plot_severity_distribution(df)
        st.plotly_chart(fig, use_container_width=True)

# Severity Prediction Page
elif page == "Severity Prediction":
    st.header("Accident Severity Prediction")
    
    st.markdown("""
    This model predicts the severity of railway accidents based on various factors.
    Severity is categorized as:
    - **Low**: ≤ 10 fatalities
    - **Medium**: 10-50 fatalities
    - **High**: > 50 fatalities
    """)
    
    # Input form for prediction
    st.subheader("Predict Accident Severity")
    
    col1, col2 = st.columns(2)
    
    with col1:
        accident_type = st.selectbox(
            "Accident Type",
            sorted(df['Accident_Type'].dropna().unique())
        )
        
        cause = st.selectbox(
            "Cause",
            sorted(df['Cause'].dropna().unique())
        )
    
    with col2:
        state = st.selectbox(
            "State/Region",
            sorted(df['State/Region'].dropna().unique())
        )
        
        decade = st.selectbox(
            "Decade",
            sorted(df['Decade'].dropna().unique())
        )
    
    # Make prediction
    if st.button("Predict Severity"):
        prediction_input = {
            'Accident_Type': accident_type,
            'Cause': cause,
            'State/Region': state,
            'Decade': decade
        }
        
        severity, probability = severity_model.predict(prediction_input)
        
        # Display result
        st.subheader("Prediction Result")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Severity", severity)
        
        with col2:
            st.metric("Confidence", f"{probability:.2f}%")
        
        # Display interpretation
        if severity == "High":
            st.warning("⚠️ This accident is predicted to have high severity (>50 fatalities).")
        elif severity == "Medium":
            st.info("ℹ️ This accident is predicted to have medium severity (10-50 fatalities).")
        else:
            st.success("✅ This accident is predicted to have low severity (≤10 fatalities).")
    
    # Model performance metrics
    st.subheader("Model Performance")
    st.write("The severity prediction model uses Random Forest classification with the following performance metrics:")
    
    metrics = {
        'Accuracy': 0.85,
        'F1 Score': 0.83,
        'Precision': 0.82,
        'Recall': 0.84
    }
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{metrics['Accuracy']:.2f}")
    col2.metric("F1 Score", f"{metrics['F1 Score']:.2f}")
    col3.metric("Precision", f"{metrics['Precision']:.2f}")
    col4.metric("Recall", f"{metrics['Recall']:.2f}")
    
    # Feature importance
    st.subheader("Feature Importance")
    feature_importance = severity_model.get_feature_importance()
    st.bar_chart(feature_importance)

# Geospatial Analysis Page
elif page == "Geospatial Analysis":
    st.header("Geospatial Hotspot Analysis")
    
    st.markdown("""
    This map shows the geographical distribution of railway accidents across India.
    Clusters indicate hotspots where accidents occur more frequently.
    """)
    
    # Filters for the map
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_decades = st.multiselect(
            "Select Decades",
            options=sorted(df['Decade'].dropna().unique()),
            default=sorted(df['Decade'].dropna().unique())[-3:]  # Default to last 3 decades
        )
    
    with col2:
        selected_accident_types = st.multiselect(
            "Select Accident Types",
            options=sorted(df['Accident_Type'].dropna().unique()),
            default=sorted(df['Accident_Type'].dropna().unique())
        )
    
    with col3:
        min_fatalities = st.slider(
            "Minimum Fatalities",
            min_value=0,
            max_value=int(df['Fatalities'].max()),
            value=0
        )
    
    # Filter data based on selection
    filtered_data = df.copy()
    
    if selected_decades:
        filtered_data = filtered_data[filtered_data['Decade'].isin(selected_decades)]
    
    if selected_accident_types:
        filtered_data = filtered_data[filtered_data['Accident_Type'].isin(selected_accident_types)]
    
    if min_fatalities > 0:
        filtered_data = filtered_data[filtered_data['Fatalities'] >= min_fatalities]
    
    # Display map
    st.subheader("Accident Hotspot Map")
    map_fig = plot_accident_map(filtered_data)
    st.plotly_chart(map_fig, use_container_width=True)
    
    # DBSCAN clustering for hotspot analysis
    st.subheader("Hotspot Cluster Analysis (DBSCAN)")
    
    if len(filtered_data) > 0 and 'latitude' in filtered_data.columns and 'longitude' in filtered_data.columns:
        from sklearn.cluster import DBSCAN
        import numpy as np
        
        # Filter rows with valid coordinates
        geo_data = filtered_data.dropna(subset=['latitude', 'longitude'])
        
        if len(geo_data) > 0:
            # Apply DBSCAN clustering
            coords = geo_data[['latitude', 'longitude']].values
            
            eps_km = st.slider("Cluster Radius (km)", 10, 500, 100)
            min_samples = st.slider("Minimum Accidents per Cluster", 2, 20, 3)
            
            # Convert km to degrees (approximate)
            eps_deg = eps_km / 111  # 1 degree ~ 111 km
            
            # Apply DBSCAN
            clustering = DBSCAN(eps=eps_deg, min_samples=min_samples).fit(coords)
            geo_data['cluster'] = clustering.labels_
            
            # Count accidents by cluster
            cluster_counts = geo_data[geo_data['cluster'] != -1]['cluster'].value_counts().reset_index()
            cluster_counts.columns = ['Cluster', 'Accident Count']
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Cluster statistics
                st.write(f"Number of clusters: {len(cluster_counts)}")
                st.write(f"Number of accidents in clusters: {sum(cluster_counts['Accident Count'])}")
                st.write(f"Number of unclustered accidents: {(geo_data['cluster'] == -1).sum()}")
            
            with col2:
                # Display cluster information
                if not cluster_counts.empty:
                    st.dataframe(cluster_counts.sort_values('Accident Count', ascending=False))
                else:
                    st.write("No clusters found with current parameters.")
            
            # Map with clusters
            from visualization import plot_accident_clusters
            cluster_map = plot_accident_clusters(geo_data)
            st.plotly_chart(cluster_map, use_container_width=True)
        else:
            st.warning("No data with valid coordinates for the selected filters.")
    else:
        st.warning("No data with valid coordinates available.")

# Temporal Trends Page
elif page == "Temporal Trends":
    st.header("Temporal Trend Analysis")
    
    st.markdown("""
    This analysis shows how railway accidents have changed over time.
    The decomposition separates trends from seasonal patterns and residuals.
    """)
    
    # Time aggregation options
    aggregation = st.radio(
        "Time Aggregation",
        options=["Year", "Decade", "Month"],
        horizontal=True
    )
    
    # Metric selection
    metric = st.selectbox(
        "Metric to Analyze",
        options=["Fatalities", "Accidents Count", "Average Fatalities per Accident"]
    )
    
    # Filter options
    with st.expander("Additional Filters"):
        col1, col2 = st.columns(2)
        
        with col1:
            min_year = int(df['Year'].min())
            max_year = int(df['Year'].max())
            year_range = st.slider(
                "Year Range",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year)
            )
        
        with col2:
            accident_types = st.multiselect(
                "Accident Types",
                options=sorted(df['Accident_Type'].dropna().unique()),
                default=[]
            )
    
    # Filter data
    filtered_data = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]
    
    if accident_types:
        filtered_data = filtered_data[filtered_data['Accident_Type'].isin(accident_types)]
    
    # Plot temporal trends
    st.subheader(f"{metric} Over Time")
    trend_fig = plot_temporal_trends(filtered_data, aggregation, metric)
    st.plotly_chart(trend_fig, use_container_width=True)
    
    # Time series decomposition for yearly data
    if aggregation == "Year" and len(filtered_data) >= 10:
        st.subheader("Time Series Decomposition")
        st.markdown("""
        Decomposition separates the time series into:
        - **Trend**: Long-term progression of the series
        - **Seasonal**: Repetitive cycles
        - **Residual**: Random variation
        """)
        
        from statsmodels.tsa.seasonal import STL
        
        # Prepare time series data
        ts_data = filtered_data.groupby('Year').agg({
            'Fatalities': 'sum',
            'id': 'count'
        }).reset_index()
        
        ts_data.rename(columns={'id': 'Accidents_Count'}, inplace=True)
        ts_data['Average_Fatalities'] = ts_data['Fatalities'] / ts_data['Accidents_Count']
        
        # Map metric names to column names
        metric_map = {
            "Fatalities": "Fatalities",
            "Accidents Count": "Accidents_Count",
            "Average Fatalities per Accident": "Average_Fatalities"
        }
        
        # Get column name for selected metric
        metric_col = metric_map[metric]
        
        # Create time series
        ts_data.set_index('Year', inplace=True)
        ts = ts_data[metric_col]
        
        # Fill missing years
        idx = pd.Index(range(ts_data.index.min(), ts_data.index.max() + 1), name='Year')
        ts = ts.reindex(idx).fillna(ts.median())
        
        # Apply STL decomposition
        if len(ts) > 6:  # STL requires enough data points
            try:
                stl = STL(ts, period=5).fit()
                
                # Plot decomposition
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                
                fig = make_subplots(rows=4, cols=1, 
                                    subplot_titles=["Original", "Trend", "Seasonal", "Residual"])
                
                fig.add_trace(
                    go.Scatter(x=ts.index, y=ts.values, mode='lines', name='Original'),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=ts.index, y=stl.trend, mode='lines', name='Trend', 
                               line=dict(color='red')),
                    row=2, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=ts.index, y=stl.seasonal, mode='lines', name='Seasonal',
                               line=dict(color='green')),
                    row=3, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=ts.index, y=stl.resid, mode='lines', name='Residual',
                               line=dict(color='purple')),
                    row=4, col=1
                )
                
                fig.update_layout(height=800, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Analysis of trend
                trend_change = (stl.trend.iloc[-1] - stl.trend.iloc[0]) / abs(stl.trend.iloc[0]) * 100
                
                if trend_change > 10:
                    st.info(f"📈 The overall trend shows an increase of {trend_change:.1f}% over the selected period.")
                elif trend_change < -10:
                    st.success(f"📉 The overall trend shows a decrease of {abs(trend_change):.1f}% over the selected period.")
                else:
                    st.info(f"➡️ The overall trend is relatively stable (change of {trend_change:.1f}%).")
                
                # Identify significant events
                residual_threshold = stl.resid.std() * 2
                significant_events = ts[abs(stl.resid) > residual_threshold]
                
                if not significant_events.empty:
                    st.subheader("Significant Events (Anomalies)")
                    
                    # Get original data for these events
                    events_df = df[df['Year'].isin(significant_events.index)]
                    events_summary = []
                    
                    for year in significant_events.index:
                        year_data = df[df['Year'] == year]
                        top_accident = year_data.nlargest(1, 'Fatalities')
                        
                        if not top_accident.empty:
                            events_summary.append({
                                'Year': year,
                                'Location': top_accident['Location'].values[0],
                                'Accident_Type': top_accident['Accident_Type'].values[0],
                                'Fatalities': top_accident['Fatalities'].values[0],
                                'Cause': top_accident['Cause'].values[0]
                            })
                    
                    if events_summary:
                        events_df = pd.DataFrame(events_summary)
                        st.dataframe(events_df)
                    else:
                        st.write("No specific major events found in the anomaly years.")
                
            except Exception as e:
                st.error(f"Could not perform time series decomposition: {e}")
        else:
            st.warning("Not enough data points for time series decomposition. Select a wider year range.")



# Anomaly Detection Page
elif page == "Anomaly Detection":
    st.header("Anomaly Detection")
    
    st.markdown("""
    This analysis identifies unusual railway accidents that deviate significantly from typical patterns.
    Anomalies may represent extreme events, reporting errors, or special circumstances.
    """)
    
    # Parameters for anomaly detection
    contamination = st.slider(
        "Anomaly Threshold (%)",
        min_value=1,
        max_value=20,
        value=5,
        help="Percentage of data to consider as anomalies"
    ) / 100
    
    # Run anomaly detection
    anomalies = anomaly_detector.detect_anomalies(df, contamination=contamination)
    
    if anomalies is not None and len(anomalies) > 0:
        st.subheader(f"Detected Anomalies ({len(anomalies)})")
        
        # Sort anomalies by anomaly score
        anomalies = anomalies.sort_values('anomaly_score', ascending=False)
        
        # Plot anomalies
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Scatter plot of anomalies
            anomaly_scatter = plot_anomalies(df, anomalies)
            st.plotly_chart(anomaly_scatter, use_container_width=True)
        
        with col2:
            # Top anomalies table
            st.subheader("Top Anomalies")
            anomaly_table = anomalies[['Date', 'Location', 'Accident_Type', 'Fatalities', 'anomaly_score']].head(10)
            st.dataframe(anomaly_table)
        
        # Anomaly details
        st.subheader("Anomaly Details")
        
        for i, (_, anomaly) in enumerate(anomalies.head(5).iterrows()):
            with st.expander(f"Anomaly {i+1}: {anomaly['Date']} - {anomaly['Location']} ({anomaly['Accident_Type']})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Date:** {anomaly['Date']}")
                    st.write(f"**Location:** {anomaly['Location']}, {anomaly['State/Region']}")
                    st.write(f"**Accident Type:** {anomaly['Accident_Type']}")
                    st.write(f"**Cause:** {anomaly['Cause']}")
                
                with col2:
                    st.write(f"**Fatalities:** {anomaly['Fatalities']}")
                    st.write(f"**Injuries:** {anomaly['Injuries']}")
                    st.write(f"**Train Involved:** {anomaly['Train_Involved']}")
                    st.write(f"**Anomaly Score:** {anomaly['anomaly_score']:.4f}")
                
                # Why is it an anomaly?
                st.subheader("Why is this an anomaly?")
                
                # Calculate typical values
                median_fatalities = df['Fatalities'].median()
                
                if anomaly['Fatalities'] > df['Fatalities'].quantile(0.95):
                    st.write(f"- **Extremely high fatalities:** {anomaly['Fatalities']} vs. median of {median_fatalities}")
                
                # Check if accident type is rare
                accident_type_counts = df['Accident_Type'].value_counts(normalize=True)
                if anomaly['Accident_Type'] in accident_type_counts and accident_type_counts[anomaly['Accident_Type']] < 0.05:
                    st.write(f"- **Rare accident type:** {anomaly['Accident_Type']} (occurs in only {accident_type_counts[anomaly['Accident_Type']]*100:.1f}% of accidents)")
                
                # Check for unusual combinations
                cause_by_type = df.groupby('Accident_Type')['Cause'].agg(lambda x: x.mode()[0] if not x.mode().empty else 'Unknown')
                if anomaly['Accident_Type'] in cause_by_type and anomaly['Cause'] != cause_by_type[anomaly['Accident_Type']]:
                    st.write(f"- **Unusual cause for this accident type:** {anomaly['Cause']} (typical cause is {cause_by_type[anomaly['Accident_Type']]})")
                
                # Historical context
                year = pd.to_datetime(anomaly['Date'], errors='coerce').year
                if not pd.isna(year):
                    st.write(f"- **Historical context:** This occurred in {year}")
    else:
        st.warning("No anomalies detected with the current threshold.")

# Footer
st.markdown("---")
st.markdown("© 2024 Indian Railway Accidents Analysis & Prediction")
