import pandas as pd
from openTransportKNN import OpenTransportKNN

df = pd.read_csv('./original_data/nox_no2.csv')

sampled_df = df.sample(n=1000, random_state=42)

pred = OpenTransportKNN()

new_col = []
a = 0
for _, r in sampled_df.iterrows():
    lat = r['latitude']
    lon = r['longitude']
    result = pred.predict(lon, lat, 0.003)
    new_col.append([float(x) for x in result[0]])
    a += 1
    print(a)
sampled_df['indexCarreteres'] = new_col

sampled_df.to_csv('./original_data/sisisi.csv', index=False)