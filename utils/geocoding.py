import pandas as pd
import numpy as np
import time
import os
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

# Cache for geocoded locations to avoid repeated API calls
geocode_cache = {}

def geocode_location(location, state=None, country="India"):
    """
    Geocode a location to get its latitude and longitude.
    
    Args:
        location: Name of the location
        state: State/region of the location
        country: Country (default: India)
        
    Returns:
        (latitude, longitude) tuple or None if geocoding fails
    """
    if pd.isna(location) or location == '':
        return None
    
    # Create a cache key
    if pd.isna(state) or state == '':
        cache_key = f"{location}, {country}"
    else:
        cache_key = f"{location}, {state}, {country}"
    
    # Check if result is in cache
    if cache_key in geocode_cache:
        return geocode_cache[cache_key]
    
    # Create geocoder
    geolocator = Nominatim(user_agent="railway_accidents_analysis")
    
    # Try to geocode
    try:
        # First try with both location and state
        if not pd.isna(state) and state != '':
            query = f"{location}, {state}, {country}"
            geocode_result = geolocator.geocode(query)
            
            # If that fails, try with location only
            if geocode_result is None:
                query = f"{location}, {country}"
                geocode_result = geolocator.geocode(query)
        else:
            query = f"{location}, {country}"
            geocode_result = geolocator.geocode(query)
        
        # If geocoding was successful, return and cache the coordinates
        if geocode_result:
            coords = (geocode_result.latitude, geocode_result.longitude)
            geocode_cache[cache_key] = coords
            return coords
        else:
            # If geocoding failed, try with state only
            if not pd.isna(state) and state != '':
                query = f"{state}, {country}"
                geocode_result = geolocator.geocode(query)
                
                if geocode_result:
                    coords = (geocode_result.latitude, geocode_result.longitude)
                    geocode_cache[cache_key] = coords
                    return coords
            
            geocode_cache[cache_key] = None
            return None
            
    except (GeocoderTimedOut, GeocoderUnavailable):
        # If there's a timeout or the service is unavailable, return None
        geocode_cache[cache_key] = None
        return None

def geocode_locations(df):
    """
    Geocode all locations in the dataset.
    
    Args:
        df: Pandas DataFrame containing the railway accidents data
        
    Returns:
        DataFrame with latitude and longitude columns
    """
    # Make a copy of the dataframe
    df = df.copy()
    
    # Create empty latitude and longitude columns
    if 'latitude' not in df.columns:
        df['latitude'] = np.nan
    
    if 'longitude' not in df.columns:
        df['longitude'] = np.nan
    
    # For each row with a missing lat/long, try to geocode
    for idx, row in df[df['latitude'].isna() | df['longitude'].isna()].iterrows():
        location = row['Location']
        state = row['State/Region']
        
        # Skip if location is missing
        if pd.isna(location) or location == '':
            continue
        
        # Geocode the location
        coords = geocode_location(location, state)
        
        # Update the dataframe if geocoding was successful
        if coords:
            df.at[idx, 'latitude'] = coords[0]
            df.at[idx, 'longitude'] = coords[1]
            
            # Delay to avoid hitting API rate limits
            time.sleep(0.1)
        else:
            # If location geocoding failed, try state-level geocoding
            if not pd.isna(state) and state != '':
                state_coords = geocode_location(None, state)
                
                if state_coords:
                    df.at[idx, 'latitude'] = state_coords[0]
                    df.at[idx, 'longitude'] = state_coords[1]
                    
                    # Delay to avoid hitting API rate limits
                    time.sleep(0.1)
    
    # Provide default coordinates for India for any remaining missing values
    # This is for visualization purposes only
    default_lat, default_lng = 20.5937, 78.9629  # Center of India
    
    # Fill missing values with defaults (with small random offsets to avoid overlapping)
    mask = df['latitude'].isna() | df['longitude'].isna()
    n_missing = mask.sum()
    
    if n_missing > 0:
        # Generate random offsets
        lat_offsets = np.random.uniform(-3, 3, n_missing)
        lng_offsets = np.random.uniform(-3, 3, n_missing)
        
        # Apply offsets to default coordinates
        df.loc[mask, 'latitude'] = default_lat + lat_offsets
        df.loc[mask, 'longitude'] = default_lng + lng_offsets
    
    return df
