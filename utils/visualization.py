import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_accident_map(df):
    """
    Create a map visualization of accident locations.
    
    Args:
        df: DataFrame with latitude and longitude columns
        
    Returns:
        Plotly figure object
    """
    # Filter rows with valid coordinates
    geo_df = df.dropna(subset=['latitude', 'longitude'])
    
    if len(geo_df) == 0:
        # Create empty map centered on India
        fig = px.scatter_mapbox(
            lat=[20.5937],
            lon=[78.9629],
            zoom=4,
            height=600
        )
        fig.update_layout(
            mapbox_style="open-street-map",
            margin={"r": 0, "t": 0, "l": 0, "b": 0}
        )
        return fig
    
    # Create hover text
    geo_df['hover_text'] = geo_df.apply(
        lambda row: f"<b>{row['Location']}, {row['State/Region']}</b><br>" +
                   f"Date: {row['Date']}<br>" +
                   f"Accident Type: {row['Accident_Type']}<br>" +
                   f"Cause: {row['Cause']}<br>" +
                   f"Fatalities: {row['Fatalities']}<br>" +
                   f"Injuries: {row['Injuries']}<br>" +
                   f"Train: {row['Train_Involved']}",
        axis=1
    )
    
    # Create map
    fig = px.scatter_mapbox(
        geo_df,
        lat="latitude",
        lon="longitude",
        color="Fatalities",
        size="Fatalities",
        color_continuous_scale="Reds",
        size_max=15,
        zoom=4,
        hover_name="Location",
        hover_data=["Date", "Accident_Type", "Fatalities", "Injuries"],
        height=600,
        opacity=0.8
    )
    
    fig.update_layout(
        mapbox_style="open-street-map",
        margin={"r": 0, "t": 0, "l": 0, "b": 0}
    )
    
    return fig

def plot_accident_clusters(df):
    """
    Create a map visualization of accident clusters.
    
    Args:
        df: DataFrame with cluster column
        
    Returns:
        Plotly figure object
    """
    # Create a color map for clusters
    clusters = sorted(df['cluster'].unique())
    colors = px.colors.qualitative.Bold
    
    # Create map
    fig = go.Figure()
    
    # Add a scatter trace for each cluster
    for i, cluster in enumerate(clusters):
        if cluster == -1:
            # Noise points (not in any cluster)
            cluster_df = df[df['cluster'] == cluster]
            fig.add_trace(go.Scattermapbox(
                lat=cluster_df['latitude'],
                lon=cluster_df['longitude'],
                mode='markers',
                marker=dict(
                    size=8,
                    color='gray',
                    opacity=0.5
                ),
                text=cluster_df['Location'],
                hoverinfo='text',
                name='Unclustered'
            ))
        else:
            # Cluster points
            cluster_df = df[df['cluster'] == cluster]
            fig.add_trace(go.Scattermapbox(
                lat=cluster_df['latitude'],
                lon=cluster_df['longitude'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=colors[i % len(colors)],
                    opacity=0.8
                ),
                text=cluster_df.apply(
                    lambda row: f"{row['Location']}: {int(row['Fatalities'])} fatalities",
                    axis=1
                ),
                hoverinfo='text',
                name=f'Cluster {cluster} ({len(cluster_df)} accidents)'
            ))
    
    # Update layout
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(
            center=dict(lat=22, lon=82),
            zoom=4
        ),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def plot_temporal_trends(df, aggregation, metric):
    """
    Create a visualization of accident trends over time.
    
    Args:
        df: DataFrame with temporal features
        aggregation: Time aggregation level (Year, Decade, Month)
        metric: Metric to visualize
        
    Returns:
        Plotly figure object
    """
    # Aggregate data
    if aggregation == "Year":
        time_column = "Year"
    elif aggregation == "Decade":
        time_column = "Decade"
    else:  # Month
        time_column = "Month"
    
    # Prepare aggregated data based on metric
    if metric == "Fatalities":
        agg_df = df.groupby(time_column)['Fatalities'].sum().reset_index()
        y_column = "Fatalities"
        title = f"Total Fatalities by {aggregation}"
    elif metric == "Accidents Count":
        agg_df = df.groupby(time_column).size().reset_index(name='Accidents_Count')
        y_column = "Accidents_Count"
        title = f"Number of Accidents by {aggregation}"
    else:  # Average Fatalities per Accident
        total_fatalities = df.groupby(time_column)['Fatalities'].sum()
        accident_counts = df.groupby(time_column).size()
        agg_df = pd.DataFrame({
            time_column: total_fatalities.index,
            'Average_Fatalities': total_fatalities.values / accident_counts.values
        })
        y_column = "Average_Fatalities"
        title = f"Average Fatalities per Accident by {aggregation}"
    
    # Sort by time
    if aggregation == "Year" or aggregation == "Decade":
        agg_df = agg_df.sort_values(time_column)
    
    # Create figure
    fig = px.line(
        agg_df,
        x=time_column,
        y=y_column,
        markers=True,
        title=title
    )
    
    # Add a trend line (moving average)
    if len(agg_df) > 5 and (aggregation == "Year" or aggregation == "Decade"):
        window = min(5, len(agg_df) // 2)
        agg_df['MA'] = agg_df[y_column].rolling(window=window, center=True).mean()
        
        fig.add_trace(
            go.Scatter(
                x=agg_df[time_column],
                y=agg_df['MA'],
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                name=f'{window}-point Moving Average'
            )
        )
    
    # Update layout
    fig.update_layout(
        xaxis_title=aggregation,
        yaxis_title=metric,
        hovermode="x unified"
    )
    
    return fig

def plot_severity_distribution(df):
    """
    Create a visualization of the severity distribution.
    
    Args:
        df: DataFrame with Severity column
        
    Returns:
        Plotly figure object
    """
    severity_counts = df['Severity'].value_counts().reset_index()
    severity_counts.columns = ['Severity', 'Count']
    
    # Ensure correct order of severity levels
    severity_order = ['Low', 'Medium', 'High']
    severity_counts['Severity'] = pd.Categorical(
        severity_counts['Severity'],
        categories=severity_order,
        ordered=True
    )
    severity_counts = severity_counts.sort_values('Severity')
    
    # Create figure
    fig = px.bar(
        severity_counts,
        x='Severity',
        y='Count',
        color='Severity',
        color_discrete_map={
            'Low': 'green',
            'Medium': 'orange',
            'High': 'red'
        },
        text='Count'
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Severity Level",
        yaxis_title="Number of Accidents",
        showlegend=False
    )
    
    # Add data labels
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    
    return fig

def plot_accident_types(df):
    """
    Create a visualization of accident types.
    
    Args:
        df: DataFrame with Accident_Type column
        
    Returns:
        Plotly figure object
    """
    # Count accidents by type
    type_counts = df['Accident_Type'].value_counts().reset_index()
    type_counts.columns = ['Accident_Type', 'Count']
    
    # Sort by count and take top 10
    type_counts = type_counts.sort_values('Count', ascending=False).head(10)
    
    # Create figure
    fig = px.bar(
        type_counts,
        x='Count',
        y='Accident_Type',
        orientation='h',
        text='Count'
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Number of Accidents",
        yaxis_title="Accident Type",
        yaxis=dict(autorange="reversed")  # Reverse y-axis to show highest count at top
    )
    
    # Add data labels
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    
    return fig



def plot_anomalies(df, anomalies):
    """
    Visualize anomalies in a scatter plot.
    
    Args:
        df: Original DataFrame
        anomalies: DataFrame with anomalies
        
    Returns:
        Plotly figure object
    """
    # Create a copy of the original data
    plot_df = df[['Fatalities', 'Injuries', 'Date', 'Location', 'Accident_Type']].copy()
    
    # Add anomaly flag
    plot_df['is_anomaly'] = False
    plot_df.loc[plot_df.index.isin(anomalies.index), 'is_anomaly'] = True
    
    # Create scatter plot
    fig = px.scatter(
        plot_df,
        x='Fatalities',
        y='Injuries',
        color='is_anomaly',
        size='Fatalities',
        hover_name='Location',
        hover_data=['Date', 'Accident_Type'],
        color_discrete_map={
            False: 'blue',
            True: 'red'
        },
        labels={
            'is_anomaly': 'Anomaly'
        }
    )
    
    # Update layout
    fig.update_layout(
        title='Anomaly Detection: Fatalities vs. Injuries',
        xaxis_title='Fatalities',
        yaxis_title='Injuries',
        height=500
    )
    
    return fig
