import pandas as pd
import netCDF4 as nc
import os
import numpy as np
import math
from netCDF4 import num2date, date2num

CONVERSION_CONSTANT = 1912.5

def caliope_model_victor_predict(date, lat, lon):
    filenames = os.listdir("original_data/NO2")
    filename = get_filename_for_date(date, filenames)
    if filename is None:
        print("No file found for the given date")
        return (0,0)

    dataset = nc.Dataset(f"original_data/NO2/{filename}", mode='r')
    # dataset = nc.Dataset("/home/vicbentu/BitsxMarato/original_data/HERMESv3_Delta_500m_SNAP07_CAT_2021082800.nc", mode='r')


    # print (dataset.variables.keys())
    # var = 'lon'
    # print(dataset.variables[var])
    # # print(dataset.variables[var][:])
    # print()
    # var = 'lat'
    # print(dataset.variables[var])
    # # print(dataset.variables[var][:])
    # print()
    # # var = 'x'
    # # print(dataset.variables[var])
    # # print(dataset.variables[var][:])
    # # print()
    # # var = 'sconcno2'
    # # print(dataset.variables[var])
    # # print(dataset.variables[var][:])
    # # print()

    # Extract lat, lon, time, sconcno2 variables
    lats = dataset.variables['lat'][:]  # shape (y, x)
    lons = dataset.variables['lon'][:]  # shape (y, x)
    hour = date.hour

    # Calculate distance
    lat_diff = lats - lat
    lon_diff = lons - lon
    dist = np.sqrt(lat_diff**2 + lon_diff**2)

    # # Print the smallest distance of the absolute value of the differences
    # print("Smallest distance:", np.min(abs(dist)))
    # print("Smallest lon:", np.min(abs(lon_diff)))
    # print("Smallest lat:", np.min(abs(lat_diff)))
    
    # y = lat, x = lon
    y_idx, x_idx = np.unravel_index(np.argmin(dist), dist.shape)
    nonweighted_mean = dataset.variables['sconcno2'][hour, 0, y_idx, x_idx] * CONVERSION_CONSTANT
    # print("*"*50)
    # print((lats[y_idx-1, x_idx-1], lons[y_idx-1, x_idx-1]))
    # print((lats[y_idx, x_idx], lons[y_idx, x_idx]))
    # print((lats[y_idx+1, x_idx+1], lons[y_idx+1, x_idx+1]))

    offsets = [(-1,-1), (0,-1), (1,-1),
               (-1, 0), (0, 0), (1, 0),
               (-1, 1), (0, 1), (1, 1)]

    values = []
    weights = []

    for (dx, dy) in offsets:
        yy = y_idx + dy
        xx = x_idx + dx
        cell_value = dataset.variables['sconcno2'][hour, 0, yy, xx] * CONVERSION_CONSTANT
        # print(cell_value)

        real_lat = lats[y_idx, x_idx]
        real_lon = lons[y_idx, x_idx]

        cell_lat = lats[yy, xx]
        cell_lon = lons[yy, xx]

        dist_lat = abs(real_lat - cell_lat)
        dist_lon = abs(real_lon - cell_lon)

        # Calculate distance from center cell in grid terms
        dist = math.sqrt(dist_lat*dist_lat + dist_lon*dist_lon)

        values.append(cell_value)
        weights.append(dist)

    # print(weights)
    max_weight = max(weights)
    weights = [max_weight - w for w in weights]
    # print(weights)

    dataset.close()

    # Compute weighted average
    weighted_mean = sum(v * w for v, w in zip(values, weights)) / sum(weights)
    # print("Weighted mean:", weighted_mean)   
    # print("Non-weighted mean:", nonweighted_mean)
    # print("Difference:", nonweighted_mean - weighted_mean)

    variance = np.var(values)
    range_val = np.max(values) - np.min(values)
    if range_val != 0:
        normalized_variance = variance / (range_val ** 2)
    else:
        normalized_variance = 0 

    return (weighted_mean, normalized_variance) 


def get_filename_for_date(date, file_names):
    """
    Get the filename corresponding to a given date.
    
    Args:
        date (datetime): The date to match against the filenames.
        file_names (list): List of filenames in the format PREFIX_YYYYMMDDHH.nc.
    
    Returns:
        str: The filename containing the date, or None if no match is found.
    """
    # Prepare a sorted list of timestamps and corresponding filenames
    timestamps = []
    for file in file_names:
        parts = file.split('_')
        if len(parts) > 1 and parts[1].endswith('.nc'):
            try:
                # Extract the timestamp part (YYYYMMDDHH)
                timestamp_str = parts[1].split('.')[0]
                timestamp = pd.to_datetime(timestamp_str, format='%Y%m%d%H')
                timestamps.append((timestamp, file))
            except ValueError:
                continue  # Skip files with invalid timestamps
    
    # Sort the timestamps
    timestamps.sort(key=lambda x: x[0])  # Sort by datetime
    
    # Find the correct file for the given date
    for i in range(len(timestamps) - 1):
        if timestamps[i][0] <= date < timestamps[i + 1][0]:
            return timestamps[i][1]
    
    # If the date is after the last file's timestamp
    if timestamps and date >= timestamps[-1][0]:
        return timestamps[-1][1]
    
    # Return None if no match is found
    return None

print (caliope_model_victor_predict(pd.to_datetime("2023-07-01 00:00:00"), 41.67940891159442, 1.6418787175324554))