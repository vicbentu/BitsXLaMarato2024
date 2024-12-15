import netCDF4 as nc
import csv
import os
import re

filenames = os.listdir("./original_dataaux/contaminacio")
dataset = nc.Dataset(f"./original_dataaux/contaminacio/{filenames[0]}", mode='r')

nox_no2 = dataset.variables['nox_no2'][:]
times = dataset.variables['time'][:]
latitudes = dataset.variables['lat'][:]
longitudes = dataset.variables['lon'][:]

lat_mask=[]
lon_mask=[]
t_mask=[]



match = re.search(r'_(\d{4})(\d{2})(\d{2})', filenames[0])
year, month, day = match.groups()

with open('nox_no2.csv', mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['time', 'latitude', 'longitude', 'nox_no2'])
    for t in range(nox_no2.shape[0]):
        for lat_idx in range(nox_no2.shape[2]):
            for lon_idx in range(nox_no2.shape[3]):
                value = nox_no2[t][0][lat_idx][lon_idx]
                if value != 0.0:
                    t_mask.append(t)
                    lat_mask.append(lat_idx)
                    lon_mask.append(lon_idx)
                    date_string = f"2023-{month}-{day} {t:02d}:00:00"
                    writer.writerow([date_string, latitudes[lat_idx], longitudes[lon_idx], value])


for filename in filenames[1:]:
    print(f"Processing {filename}")
    dataset = nc.Dataset(f"./original_dataaux/contaminacio/{filename}", mode='r')
    match = re.search(r'_(\d{4})(\d{2})(\d{2})', filenames[0])
    year, month, day = match.groups()
    nox_no2 = dataset.variables['nox_no2'][:]
    times = dataset.variables['time'][:]
    latitudes = dataset.variables['lat'][:]
    longitudes = dataset.variables['lon'][:]
    i=0
    k =1
    with open('nox_no2.csv', mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['time', 'latitude', 'longitude', 'nox_no2'])
        
        for i in range(len(t_mask)):
            value = nox_no2[t_mask[i]][0][lat_mask[i]][lon_mask[i]]
            date_string = f"2023-{month}-{day} {t_mask[i]:02d}:00:00"
            i+=1
            if i==1000*k:
                print(date_string)
                k+=1
            writer.writerow([date_string, latitudes[lat_mask], longitudes[lon_mask], value])
                    