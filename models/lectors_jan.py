import pandas as pd
import math
import numpy as np

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth.
    
    Parameters:
        lat1, lon1: Latitude and longitude of the first point in decimal degrees.
        lat2, lon2: Latitude and longitude of the second point in decimal degrees.
        
    Returns:
        Distance in meters.
    """
    # Earth radius in meters
    R = 6371000  
    
    # Convert latitude and longitude from degrees to radians
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
    
    # Compute differences
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad
    
    # Haversine formula
    a = math.sin(delta_lat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    # Compute the distance
    distance = R * c
    return distance

class tuMama:
    def __init__(self):
        self.df = pd.read_csv('../original_dataaux/AMS_Observacions/XVPCA_info_sconcno2_2023.csv')
        self.radar_values = pd.read_csv('../original_dataaux/AMS_Observacions/gene_sconcno2_2023_xvpca_emep_port.csv')


    def get_nearest_stations_calc_value(self, target_date,lat,lon,threshold,unneded_types):
        newdf=self.df.copy()
        newdf["distance"] = newdf.apply(lambda row: haversine(lat, lon, row["lat"], row["lon"]), axis=1)
        radar_loc = newdf[newdf["distance"] < threshold]
        radar_values = pd.read_csv('../original_dataaux/AMS_Observacions/gene_sconcno2_2023_xvpca_emep_port.csv')

        # Filter for the specific row in df2 based on the date
        row = radar_values[radar_values['Date'] == target_date]
        if row.empty:
            raise ValueError(f"No data found for the given date: {target_date}")

        # Extract the values and filter out NA columns
        row_values = row.iloc[0, 1:]  # Exclude 'Date' column
        valid_values = row_values.dropna()  # Drop NA values

        # Match the codes in df1 with valid codes from df2
        valid_codes = valid_values.index
        matched_df1 = radar_loc[radar_loc['code'].isin(valid_codes)]
        print (matched_df1)
        if matched_df1.empty:
            raise ValueError("No matching stations with distances for the selected time.")

        # Get distances and values for the matched stations
        distances = matched_df1['distance'].values
        values = valid_values[matched_df1['code']].values
        # Perform inverse-distance weighting
        weights = 1 / distances  # Inverse distance
        interpolated_value = np.sum(weights * values) / np.sum(weights)

        types = matched_df1['type'].value_counts()

        return_array = []
        return_array.append({"type": "general", "value": interpolated_value})

        for index, _ in types.items():
            new_matched_df1 = matched_df1[matched_df1['type'] == index]
            distances = new_matched_df1['distance'].values
            values = valid_values[new_matched_df1['code']].values
            # Perform inverse-distance weighting
            weights = 1 / distances  # Inverse distance
            interpolated_value = np.sum(weights * values) / np.sum(weights)
            return_array.append({"type": index, "value": interpolated_value})

        return return_array



def compute():
    ob = tuMama()

    