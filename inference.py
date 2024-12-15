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
        # lat, lon = map(float, input("Enter latitude and longitude separated by a space: ").split())
        # Separated by space or coma
        # lat, lon = map(float, input("Enter latitude and longitude separated by a space: ").split())
        inputed = input("Enter latitude and longitude separated by a space or comma: ")
        if ',' in inputed:
            lat, lon = map(float, inputed.split(','))
        else:
            lat, lon = map(float, inputed.split())
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