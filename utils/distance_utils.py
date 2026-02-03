import pandas as pd
from math import radians, cos, sin, asin, sqrt

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 2 * asin(sqrt(a)) * 6371


def add_distance_from_location(df, location_name):
    """
    Adds distance_km and distance_rank columns
    """
    loc_df = df[df['location'].str.lower() == location_name.lower()].copy()

    loc_df['latitude'] = pd.to_numeric(loc_df['latitude'], errors='coerce')
    loc_df['longitude'] = pd.to_numeric(loc_df['longitude'], errors='coerce')
    loc_df = loc_df.dropna(subset=['latitude', 'longitude'])

    # Location centroid
    center_lat = loc_df['latitude'].mean()
    center_lon = loc_df['longitude'].mean()

    loc_df['distance_km'] = loc_df.apply(
        lambda r: haversine(center_lat, center_lon, r['latitude'], r['longitude']),
        axis=1
    )

    # Clean tiny values
    loc_df['distance_km'] = loc_df['distance_km'].round(3)
    loc_df.loc[loc_df['distance_km'] < 0.05, 'distance_km'] = 0.0

    # Distance rank (IMPORTANT)
    loc_df['distance_rank'] = loc_df['distance_km'].rank(method="dense")

    return loc_df
