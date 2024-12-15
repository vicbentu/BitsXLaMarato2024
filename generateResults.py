import torch
import pandas as pd
import torch.nn as nn
from neural_network import SimpleModel, preprocess, CONVERSION_FACTOR
from models import OpenTransportKNN, caliope_model_victor
from datetime import datetime, timedelta

model = SimpleModel() 
model.load_state_dict(torch.load("model8000.pth"))
model.eval()
openMapPredictor = OpenTransportKNN()

# Create an empty dataframe
df = pd.DataFrame(columns=["id", "date", "lat", "lon", "concentration"])

positions = [
    [40.39216, 2.009802],
    [41.11588, 1.191975],
    [41.44398, 2.237875],
    [41.32177, 2.082141]
]

try:
    for (lat, lon) in positions:
        current_time = datetime(2023, 1, 1, 0, 0)
        end_time = datetime(2023, 12, 31, 23, 0)
        while current_time <= end_time:
            _time = current_time.strftime("%Y-%m-%d %H:%M")

            input = preprocess((_time, lat, lon), openMapPredictor)
            output = model(input).item()
            # print(f"Predicted concentration at {lat}, {lon} at {_time}: {output}")

            # df = df.append({
            #     "id": len(df),
            #     "date": _time,
            #     "lat": lat,
            #     "lon": lon,
            #     "concentration": output
            # }, ignore_index=True)
            df.loc[len(df)] = [len(df)+1, _time, lat, lon, output]

            current_time += timedelta(hours=1)
            if len(df) % 500 == 0:
                print(f"Processed {len(df)} rows")

    # Save the dataframe to a CSV file
    df.to_csv("our_results.csv", index=False)
except Exception as e:
    df.to_csv("our_results.csv", index=False)
    print(f"Saved to CSV before error")
    print(e)
except KeyboardInterrupt:
    df.to_csv("our_results.csv", index=False)
    print(f"Saved to CSV before interruption")