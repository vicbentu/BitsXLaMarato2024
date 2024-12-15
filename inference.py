import torch
import pandas as pd
import torch.nn as nn
from neural_network import SimpleModel, preprocess, CONVERSION_FACTOR
from models import OpenTransportKNN, caliope_model_victor
from datetime import datetime, timedelta
import time

model = SimpleModel()
model.load_state_dict(torch.load("model4000.pth"))
model.eval()
openMapPredictor = OpenTransportKNN()

# make a loop that reads console input
while True:
    try:
        lat = float(input("Enter latitude: "))
        lon = float(input("Enter longitude: "))
        date = input("Enter date (YYYY-MM-DD HH:MM): ")
        
        my_input = preprocess((date, lat, lon), openMapPredictor)
        output = model(my_input).item()

        
        # print the output
        print(f"Predicted concentration at {lat}, {lon} at {date}:  {output}")	
    except Exception as e:
        print(e)
        time.sleep(1)
    except KeyboardInterrupt:
        break