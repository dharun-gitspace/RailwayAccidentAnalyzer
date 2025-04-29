import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Import custom modules
from data_preprocessing import preprocess_data, encode_features
from data_analysis import (
    perform_temporal_analysis, 
    create_accident_severity_heatmap,
    get_accident_statistics,
    get_state_accident_counts
)
from model import (
    train_severity_model, 
    predict_severity, 
    save_model, 
    load_model
)
from utils import (
    extract_temporal_features,
    categorize_severity
)
from anomaly_detection import (
    detect_anomalies,
    explain_anomalies
)
from association_rules import mine_association_rules
from geocoding import geocode_locations, get_state_center_coordinates

# Page configuration
st.set_page_config(
    page_title="Indian Railway Accidents Analysis",
    page_icon="🚂",
    layout="wide"
)

# App title and description
st.title("Indian Railway Accidents Analysis & Prediction")
st.markdown("""
This application analyzes Indian railway accidents data from 1902 to 2024, 
providing insights on accident severity, geospatial patterns, temporal trends, 
and predicts severity of potential future accidents.
""")

# Load and preprocess data
@st.cache_data(show_spinner=True)
def load_data():
    try:
        # Try to load the dataset
        df = pd.read_csv("indian_railway_accidents.csv")
        
        # Preprocess the data
        df = preprocess_data(df)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page",
    ["Overview", "Geospatial Analysis", "Temporal Trends", 
     "Severity Prediction", "Association Rules", "Anomaly Detection"]
)

# Load the dataset
df = load_data()

if df is None:
    st.error("Failed to load the dataset. Please check if the file exists and is in the correct format.")
    st.stop()

# Main application logic based on selected page
if page == "Overview":
    st.header("Dataset Overview")
    
    # Display dataset summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sample Data")
        st.dataframe(df.head())
    
    with col2:
        st.subheader("Dataset Summary")
        total_accidents = len(df)
        total_fatalities = df['Fatalities'].sum()
        avg_fatalities = df['Fatalities'].mean()
        total_injuries = df['Injuries'].sum()
        max_fatality_accident = df.loc[df['Fatalities'].idxmax()]
        
        st.markdown(f"**Total accidents:** {total_accidents}")
        st.markdown(f"**Total fatalities:** {int(total_fatalities)}")
        st.markdown(f"**Average fatalities per accident:** {avg_fatalities:.2f}")
        st.markdown(f"**Total injuries:** {int(total_injuries)}")
        st.markdown("**Worst accident:**")
        st.markdown(f"- Date: {max_fatality_accident['Date']}")
        st.markdown(f"- Location: {max_fatality_accident['Location']}, {max_fatality_accident['State/Region']}")
        st.markdown(f"- Fatalities: {max_fatality_accident['Fatalities']}")
        st.markdown(f"- Type: {max_fatality_accident['Accident_Type']}")
    
    # Statistics
    st.subheader("Accident Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Accidents by Type")
        accident_types = df['Accident_Type'].value_counts()
        fig = px.bar(
            x=accident_types.index, 
            y=accident_types.values,
            labels={'x': 'Accident Type', 'y': 'Count'},
            title="Accidents by Type"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Accidents by Cause")
        causes = df['Cause'].value_counts().head(10)  # Top 10 causes
        fig = px.bar(
            x=causes.index, 
            y=causes.values,
            labels={'x': 'Cause', 'y': 'Count'},
            title="Top 10 Accident Causes"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Severity distribution
    st.subheader("Accident Severity Distribution")
    df['Severity'] = df['Fatalities'].apply(categorize_severity)
    severity_counts = df['Severity'].value_counts().reset_index()
    severity_counts.columns = ['Severity', 'Count']
    
    # Order by severity
    severity_order = {'Low': 0, 'Medium': 1, 'High': 2}
    severity_counts['SortOrder'] = severity_counts['Severity'].map(severity_order)
    severity_counts = severity_counts.sort_values('SortOrder').drop('SortOrder', axis=1)
    
    fig = px.pie(
        severity_counts, 
        values='Count', 
        names='Severity',
        title="Accident Severity Distribution",
        color='Severity',
        color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'}
    )
    st.plotly_chart(fig, use_container_width=True)

elif page == "Geospatial Analysis":
    st.header("Geospatial Analysis")
    
    # Geospatial visualization
    st.subheader("Accident Hotspots")
    
    # Get coordinates for states
    state_coords = get_state_center_coordinates()
    state_accident_counts = get_state_accident_counts(df)
    
    if state_coords:
        # Create a base map centered on India
        m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
        
        # Add circles for each state with accident counts
        for state, count in state_accident_counts.items():
            if state in state_coords:
                lat, lon = state_coords[state]
                folium.Circle(
                    location=[lat, lon],
                    radius=count * 1000,  # Scale the radius based on count
                    color='crimson',
                    fill=True,
                    fill_color='crimson',
                    fill_opacity=0.6,
                    tooltip=f"{state}: {count} accidents"
                ).add_to(m)
        
        # Display the map
        folium_static(m)
    else:
        st.warning("Could not load coordinates for geospatial visualization.")
    
    # Create heatmap of accident severity by state
    st.subheader("Accident Severity Heatmap by State")
    heatmap_data = create_accident_severity_heatmap(df)
    
    if not heatmap_data.empty:
        fig = px.density_heatmap(
            heatmap_data,
            x='State/Region',
            y='Decade',
            z='Fatalities',
            title="Accident Severity Heatmap (Fatalities by State and Decade)",
            labels={'Fatalities': 'Total Fatalities'}
        )
        fig.update_layout(xaxis={'categoryorder': 'total descending'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough data to create a meaningful heatmap.")

elif page == "Temporal Trends":
    st.header("Temporal Trend Analysis")
    
    # Filter for data completeness
    st.subheader("Accident Trends Over Time")
    
    # Group by year and decade
    yearly_data, decade_data = perform_temporal_analysis(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Accidents by Year")
        fig = px.line(
            yearly_data, 
            x='Year', 
            y='Count',
            title="Number of Accidents by Year",
            labels={'Count': 'Number of Accidents', 'Year': 'Year'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Fatalities by Year")
        fig = px.line(
            yearly_data, 
            x='Year', 
            y='Fatalities',
            title="Fatalities by Year",
            labels={'Fatalities': 'Number of Fatalities', 'Year': 'Year'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("#### Accidents and Fatalities by Decade")
    fig = px.bar(
        decade_data, 
        x='Decade', 
        y=['Count', 'Fatalities'],
        title="Accidents and Fatalities by Decade",
        barmode='group',
        labels={'value': 'Count', 'Decade': 'Decade', 'variable': 'Metric'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Top accident types over time
    st.subheader("Top Accident Types Over Time")
    
    # Group by decade and accident type
    accident_types_by_decade = df.groupby(['Decade', 'Accident_Type']).size().reset_index(name='Count')
    
    # Get top accident types for each decade
    top_types_by_decade = accident_types_by_decade.sort_values(['Decade', 'Count'], ascending=[True, False])
    top_types_by_decade = top_types_by_decade.groupby('Decade').head(3)  # Top 3 per decade
    
    fig = px.bar(
        top_types_by_decade,
        x='Decade',
        y='Count',
        color='Accident_Type',
        title="Top 3 Accident Types by Decade",
        labels={'Count': 'Number of Accidents', 'Decade': 'Decade'}
    )
    st.plotly_chart(fig, use_container_width=True)

elif page == "Severity Prediction":
    st.header("Accident Severity Prediction")
    
    # Model explanation
    st.markdown("""
    This model predicts the severity of railway accidents based on features like:
    - Accident Type
    - Cause
    - State/Region
    - Decade
    
    Severity levels:
    - **Low**: Fatalities ≤ 10
    - **Medium**: 10 < Fatalities ≤ 50
    - **High**: Fatalities > 50
    """)
    
    # Train model button
    if st.button("Train Severity Prediction Model"):
        with st.spinner("Training model..."):
            model, X, features, accuracy, f1 = train_severity_model(df)
            save_model(model, features)
            
            st.success(f"Model trained successfully! Accuracy: {accuracy:.2f}, F1 Score: {f1:.2f}")
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                st.subheader("Feature Importance")
                importance_df = pd.DataFrame({
                    'Feature': [X.columns[i] for i in indices],
                    'Importance': [importances[i] for i in indices]
                })
                
                fig = px.bar(
                    importance_df,
                    x='Feature',
                    y='Importance',
                    title="Feature Importance for Severity Prediction"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Prediction form
    st.subheader("Predict Accident Severity")
    
    # Check if model exists
    model_exists = os.path.exists("severity_model.pkl") and os.path.exists("model_features.pkl")
    
    if not model_exists:
        st.warning("Please train the model first before making predictions.")
    else:
        # Load the model and features
        model, features = load_model()
        
        # Get unique values for categorical features
        accident_types = sorted(df['Accident_Type'].dropna().unique())
        causes = sorted(df['Cause'].dropna().unique())
        states = sorted(df['State/Region'].dropna().unique())
        decades = sorted(df['Decade'].dropna().unique())
        
        # Create form
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                accident_type = st.selectbox("Accident Type", accident_types)
                cause = st.selectbox("Cause", causes)
            
            with col2:
                state = st.selectbox("State/Region", states)
                decade = st.selectbox("Decade", decades)
            
            submit_button = st.form_submit_button("Predict Severity")
            
            if submit_button:
                # Make prediction
                prediction = predict_severity(model, features, {
                    'Accident_Type': accident_type,
                    'Cause': cause,
                    'State/Region': state,
                    'Decade': decade
                })
                
                # Display prediction
                severity_color = {
                    'Low': 'green',
                    'Medium': 'orange',
                    'High': 'red'
                }
                
                st.markdown(f"""
                ### Prediction Result
                The predicted severity is: 
                <span style='color:{severity_color[prediction]};font-weight:bold;font-size:24px;'>
                    {prediction}
                </span>
                """, unsafe_allow_html=True)

elif page == "Association Rules":
    st.header("Association Rule Mining")
    
    st.markdown("""
    Association rule mining discovers relationships between variables in the dataset. 
    For example, it can reveal that certain accident types are more common in specific states 
    or during certain decades.
    """)
    
    # Parameters for rule mining
    st.subheader("Set Parameters for Rule Mining")
    
    col1, col2 = st.columns(2)
    
    with col1:
        min_support = st.slider("Minimum Support", 0.01, 0.5, 0.05, 0.01,
                               help="Minimum support threshold for items to be included in rules")
    
    with col2:
        min_confidence = st.slider("Minimum Confidence", 0.1, 1.0, 0.6, 0.05,
                                  help="Minimum probability threshold for the rule to be accepted")
    
    # Features to include
    st.subheader("Select Features to Include")
    features = st.multiselect(
        "Select features for rule mining",
        ['Accident_Type', 'Cause', 'State/Region', 'Decade', 'Severity', 'Train_Involved'],
        ['Accident_Type', 'Cause', 'State/Region', 'Decade', 'Severity']
    )
    
    if len(features) < 2:
        st.warning("Please select at least 2 features for rule mining.")
    else:
        if st.button("Mine Association Rules"):
            with st.spinner("Mining association rules..."):
                rules = mine_association_rules(df, features, min_support, min_confidence)
                
                if rules is not None and not rules.empty:
                    st.success(f"Found {len(rules)} association rules.")
                    
                    # Display rules
                    st.subheader("Top Association Rules")
                    
                    # Format rules for display
                    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
                    
                    # Calculate lift and sort by it
                    rules = rules.sort_values('lift', ascending=False)
                    
                    # Display top rules
                    st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
                    
                    # Visualize top rules
                    st.subheader("Top 10 Rules by Lift")
                    
                    top_rules = rules.head(10).copy()
                    top_rules['rule'] = top_rules.apply(
                        lambda row: f"{row['antecedents']} → {row['consequents']}", axis=1
                    )
                    
                    fig = px.bar(
                        top_rules,
                        x='rule',
                        y='lift',
                        title="Top 10 Association Rules by Lift",
                        labels={'rule': 'Rule', 'lift': 'Lift (Higher is stronger)'}
                    )
                    fig.update_layout(xaxis={'categoryorder': 'total descending'})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No association rules found with the given parameters. Try lowering the support threshold.")

elif page == "Anomaly Detection":
    st.header("Anomaly Detection")
    
    st.markdown("""
    This section identifies anomalous railway accidents that deviate significantly from normal patterns.
    Anomalies could be accidents with unusually high fatalities, rare combinations of features, or other unusual characteristics.
    """)
    
    # Parameters for anomaly detection
    st.subheader("Set Parameters for Anomaly Detection")
    
    contamination = st.slider(
        "Contamination (expected proportion of anomalies)",
        0.01, 0.3, 0.05, 0.01,
        help="Higher values will flag more accidents as anomalies"
    )
    
    features = st.multiselect(
        "Select features for anomaly detection",
        ['Fatalities', 'Injuries', 'Accident_Type', 'Cause', 'State/Region', 'Decade'],
        ['Fatalities', 'Injuries', 'Accident_Type', 'State/Region', 'Decade']
    )
    
    if st.button("Detect Anomalies"):
        with st.spinner("Detecting anomalies..."):
            # Detect anomalies
            anomalies_df, anomaly_indices = detect_anomalies(df, features, contamination)
            
            if not anomalies_df.empty:
                st.success(f"Detected {len(anomalies_df)} anomalous accidents.")
                
                # Display anomalies
                st.subheader("Anomalous Accidents")
                st.dataframe(anomalies_df)
                
                # Explain anomalies
                st.subheader("Anomaly Explanations")
                
                explanations = explain_anomalies(df, anomaly_indices)
                
                for i, explanation in enumerate(explanations):
                    with st.expander(f"Anomaly #{i+1}: {explanation['summary']}"):
                        st.markdown(f"**Date:** {explanation['date']}")
                        st.markdown(f"**Location:** {explanation['location']}, {explanation['state']}")
                        st.markdown(f"**Accident Type:** {explanation['accident_type']}")
                        st.markdown(f"**Fatalities:** {explanation['fatalities']}")
                        st.markdown(f"**Cause:** {explanation['cause']}")
                        st.markdown("**Why it's anomalous:**")
                        for reason in explanation['reasons']:
                            st.markdown(f"- {reason}")
            else:
                st.warning("No anomalies detected with the current settings.")
