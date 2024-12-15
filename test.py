import pandas as pd

df = pd.read_csv("updated_data.csv")

# plot the variable nox_no2 and 
df['nox_no2'].plot()
