import pandas as pd
from datetime import datetime, timedelta

# Define the base date: 1st of January, 2023
base_date = datetime(2023, 1, 1)

try:
    # Load the CSV file with error handling for malformed lines
    data = pd.read_csv("/home/vicbentu/BitsxMarato/modDatasets/nox_no2.csv")
    
    # Update the 'time' column to reflect the new datetime format
    data['time'] = data['time'].apply(lambda x: (base_date + timedelta(hours=int(x))).strftime('%Y-%m-%d %H:%M:%S'))
    
    # Save the updated CSV
    data.to_csv("updated_data.csv", index=False)
    print("Updated CSV saved as 'updated_data.csv'")

except Exception as e:
    print(f"An error occurred: {e}")
