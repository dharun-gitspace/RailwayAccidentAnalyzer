import pandas as pd
import numpy as np
from datetime import datetime
import re

def preprocess_data(df):
    """
    Perform initial preprocessing on the railway accidents dataset.
    
    Args:
        df: Pandas DataFrame containing the railway accidents data
        
    Returns:
        Preprocessed DataFrame
    """
    # Make a copy of the dataframe to avoid modifying the original
    df = df.copy()
    
    # Add an ID column
    df['id'] = range(1, len(df) + 1)
    
    # Convert 'Not specified' to NaN
    df.replace('Not specified', np.nan, inplace=True)
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format='%m-%d-%Y')
    
    # Convert Fatalities and Injuries to numeric
    df['Fatalities'] = pd.to_numeric(df['Fatalities'], errors='coerce')
    df['Injuries'] = pd.to_numeric(df['Injuries'], errors='coerce')
    
    return df

def standardize_states(df):
    """
    Standardize state/region names to modern equivalents.
    
    Args:
        df: Pandas DataFrame containing the railway accidents data
        
    Returns:
        DataFrame with standardized state names
    """
    # Make a copy of the dataframe
    df = df.copy()
    
    # Dictionary mapping historical names to modern equivalents
    state_mapping = {
        'Madras Presidency': 'Tamil Nadu',
        'Punjab Province': 'Punjab',
        'United Provinces': 'Uttar Pradesh',
        'Bombay': 'Maharashtra',
        'Madras State': 'Tamil Nadu',
        'Madras': 'Tamil Nadu',
        'Hyderabad State': 'Telangana',
        'Hyderabad': 'Telangana',
        'Mysore state': 'Karnataka',
        'Mysore': 'Karnataka',
        'Orissa': 'Odisha',
        'Not specified': np.nan
    }
    
    # Replace state names
    df['State/Region'] = df['State/Region'].replace(state_mapping)
    
    return df

def extract_temporal_features(df):
    """
    Extract temporal features from Date column.
    
    Args:
        df: Pandas DataFrame containing the railway accidents data
        
    Returns:
        DataFrame with additional temporal features
    """
    # Make a copy of the dataframe
    df = df.copy()
    
    # Extract year, month, and decade
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    
    # Calculate decade (e.g., 1900, 1910, 1920, etc.)
    df['Decade'] = (df['Year'] // 10) * 10
    
    # Convert decade to string for better representation
    df['Decade'] = df['Decade'].apply(lambda x: f"{int(x)}s" if not pd.isna(x) else np.nan)
    
    return df

def infer_cause_from_accident_type(accident_type):
    """
    Infer cause based on accident type for missing values.
    
    Args:
        accident_type: Type of accident
        
    Returns:
        Inferred cause
    """
    if pd.isna(accident_type):
        return np.nan
    
    accident_type = str(accident_type).lower()
    
    # Mapping of accident types to causes
    cause_mapping = {
        'derailment': 'Track failure',
        'collision': 'Signaling error',
        'fire': 'Electrical fault',
        'explosion': 'Explosives accident',
        'bridge': 'Infrastructure failure',
        'bridge collapse': 'Infrastructure failure',
        'bridge accident': 'Infrastructure failure',
        'level crossing': 'Human error',
        'bombing': 'Sabotage',
        'natural disaster': 'Natural disaster',
        'crash': 'Operational error'
    }
    
    # Find the matching key in the accident type
    for key, cause in cause_mapping.items():
        if key in accident_type:
            return cause
    
    # Default cause if no match is found
    return 'Unknown'

def handle_missing_data(df):
    """
    Handle missing data in the dataset.
    
    Args:
        df: Pandas DataFrame containing the railway accidents data
        
    Returns:
        DataFrame with imputed values
    """
    # Make a copy of the dataframe
    df = df.copy()
    
    # Impute missing Cause based on Accident_Type
    cause_mask = df['Cause'].isna()
    df.loc[cause_mask, 'Cause'] = df.loc[cause_mask, 'Accident_Type'].apply(infer_cause_from_accident_type)
    
    # Group by Accident_Type and State for imputing Fatalities and Injuries
    fatality_medians = df.groupby(['Accident_Type'])['Fatalities'].median()
    injury_medians = df.groupby(['Accident_Type'])['Injuries'].median()
    
    # Impute missing Fatalities using the median for that Accident_Type
    fatality_mask = df['Fatalities'].isna()
    for idx in df[fatality_mask].index:
        accident_type = df.loc[idx, 'Accident_Type']
        if accident_type in fatality_medians and not pd.isna(fatality_medians[accident_type]):
            df.loc[idx, 'Fatalities'] = fatality_medians[accident_type]
        else:
            df.loc[idx, 'Fatalities'] = df['Fatalities'].median()
    
    # Impute missing Injuries using the median for that Accident_Type
    injury_mask = df['Injuries'].isna()
    for idx in df[injury_mask].index:
        accident_type = df.loc[idx, 'Accident_Type']
        if accident_type in injury_medians and not pd.isna(injury_medians[accident_type]):
            df.loc[idx, 'Injuries'] = injury_medians[accident_type]
        else:
            df.loc[idx, 'Injuries'] = df['Injuries'].median()
    
    # For remaining NaN values in Injuries, use a ratio based on Fatalities
    injury_mask = df['Injuries'].isna()
    fatality_injury_ratio = df[df['Fatalities'].notna() & df['Injuries'].notna()]['Injuries'].sum() / df[df['Fatalities'].notna() & df['Injuries'].notna()]['Fatalities'].sum()
    df.loc[injury_mask, 'Injuries'] = df.loc[injury_mask, 'Fatalities'] * fatality_injury_ratio
    
    # Create severity category
    df['Severity'] = pd.cut(
        df['Fatalities'], 
        bins=[0, 10, 50, float('inf')],
        labels=['Low', 'Medium', 'High'],
        right=True
    )
    
    return df
