import netCDF4 as nc
import pandas as pd
import numpy as np

# Path to your .nc file
file_path = 'original_data/contaminacio/HERMESv3_Delta_500m_SNAP07_CAT_2021010100.nc'

# Open the .nc file
dataset = nc.Dataset(file_path, 'r')

# Extract the necessary variables
time = dataset.variables['time'][:]
lat = dataset.variables['lat'][:]
lon = dataset.variables['lon'][:]
nox_no2 = dataset.variables['nox_no2'][:]

# Select specific time, lat, and lon indices
# Example: For time=0, lat=50.5, lon=10.5
time_idx = 0  # Adjust based on your needs
selected_lat = 50.5
selected_lon = 10.5

# Find the closest indices for lat and lon
lat_idx = np.abs(lat - selected_lat).argmin()
lon_idx = np.abs(lon - selected_lon).argmin()

# Extract the data for the specific time, lat, and lon
nox_no2_data = nox_no2[time_idx, :, lat_idx, lon_idx]

# Prepare data for CSV
df = pd.DataFrame({
    'Time': [time[time_idx]],
    'Latitude': [lat[lat_idx]],
    'Longitude': [lon[lon_idx]],
    'NOx_NO2': nox_no2_data.flatten()  # Flatten in case of multi-dimensional values
})

# Save to CSV
output_csv_path = 'nox_no2_output.csv'
df.to_csv(output_csv_path, index=False)

print(f"Data saved to {output_csv_path}")

# Close the dataset
dataset.close()
