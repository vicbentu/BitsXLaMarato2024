# import torch
# import torch.nn as nn
# import pandas as pd
# from math import sqrt
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score


# from models import caliope_model_victor_predict, OpenTransportKNN

# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", message="Using a target size.*")


# CONVERSION_FACTOR = 507008.0577669091

# def list_mean(lst):
#     return sum(lst) / len(lst)

# def list_std(lst):
#     mean = list_mean(lst)
#     variance = sum((x - mean) ** 2 for x in lst) / len(lst)
#     return sqrt(variance)

# def preprocess(inputs, predictor):
#     time, lat, lon = inputs
    
#     time = pd.to_datetime(time)
#     res, var = caliope_model_victor_predict(time, lat, lon)
#     road_array = predictor.predict(lat, lon)[0]

#     new_vars_array = [
#         res * var,
#         res / (var + 1e-9),
#         res - var,
#         res**2 + var**2,
#         res**2 - var**2,
#         res * var**2,
#         sum(road_array),                       # Sum of the list
#         list_mean(road_array),                 # Mean of the list
#         list_std(road_array),                  # Standard deviation of the list
#         road_array[0]**2 if len(road_array) > 0 else 0, 
#         road_array[1]**2 if len(road_array) > 1 else 0, 
#         road_array[2]**2 if len(road_array) > 2 else 0, 
#         road_array[3]**2 if len(road_array) > 3 else 0, 
#         road_array[4]**2 if len(road_array) > 4 else 0, 
#         road_array[5]**2 if len(road_array) > 5 else 0, 
#     ]
#     combined_array = [res, var] + road_array + new_vars_array
#     inputs = torch.tensor(combined_array)
#     inputs = inputs.float()
#     return inputs

# class SimpleModel(nn.Module):
#     def __init__(self, input_dim=23, hidden_dim=64, output_dim=1):
#         super(SimpleModel, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, output_dim)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# if __name__ == "__main__":
#     df = pd.read_csv("updated_data.csv")
#     model = SimpleModel()
#     initial_i = 13
#     model.load_state_dict(torch.load(f"pths/model{initial_i-1}.pth"))
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#     criterion = nn.MSELoss()
#     openMapPredictor = OpenTransportKNN()

#     train_df, test_df = train_test_split(df, test_size=0.999, random_state=42)
#     train_df, val_df = train_test_split(train_df, test_size=0.5, random_state=42)
#     y_val_tensor = torch.tensor(val_df['nox_no2'].values, dtype=torch.float32)
#     # train_df, test_df = train_test_split(df, test_size=0.99999, random_state=42)
#     # train_df, val_df = train_test_split(train_df, test_size=0.5, random_state=42)
#     # y_val_tensor = torch.tensor(val_df['nox_no2'].values, dtype=torch.float32)
    

#     # print size of train_df
#     print("-"*100)
#     print(f"Size of train_df: {train_df.shape[0]}")
#     print(f"Size of test_df: {test_df.shape[0]}")
#     print(f"Size of val_df: {val_df.shape[0]}")


#     for epoch in range(100):
#         # for inputs, target in dataloader:
#         print(f"------------------------------- EPOCH {epoch} -------------------------------")
#         for index, row in train_df.iterrows():
#             time, lat, lon, no2 = row['time'], row['latitude'], row['longitude'], row['nox_no2']
#             inputs = (time, lat, lon)
#             inputs = preprocess(inputs, openMapPredictor)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             target = row['nox_no2'] * CONVERSION_FACTOR
#             target = torch.tensor(target, dtype=torch.float32)
#             loss = criterion(outputs, target)
#             loss.backward()
#             optimizer.step()
        

#         model.eval()
#         val_loss = 0.0
#         y_trues = []
#         y_preds = []
#         with torch.no_grad():
#             for index, row in val_df.iterrows():
#                 time, lat, lon, no2 = row['time'], row['latitude'], row['longitude'], row['nox_no2']
#                 inputs = (time, lat, lon)
#                 inputs = preprocess(inputs, openMapPredictor)
#                 outputs = model(inputs).unsqueeze(0)
#                 target = row['nox_no2'] * CONVERSION_FACTOR
#                 target = torch.tensor(target, dtype=torch.float32)
#                 val_loss += criterion(outputs, target).item()
                
#                 y_trues.append(target.item())
#                 y_preds.append(outputs.squeeze().item())

#         val_loss /= len(val_df)
#         print(f"Validation Loss: {val_loss:.4f}")
#         # Print r-squared
#         r2 = r2_score(y_trues, y_preds)
#         print(f"R-squared: {r2:.4f}")
            

#         torch.save(model.state_dict(), f'pths/model{epoch+initial_i}.pth')

#     torch.save(model.state_dict(), 'model.pth')





import torch
import json
import torch.nn as nn
import pandas as pd
from math import sqrt, isnan
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time

from models import caliope_model_victor_predict, OpenTransportKNN

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Using a target size.*")


CONVERSION_FACTOR = 507008.0577669091

def list_mean(lst):
    return sum(lst) / len(lst)

def list_std(lst):
    mean = list_mean(lst)
    variance = sum((x - mean) ** 2 for x in lst) / len(lst)
    return sqrt(variance)

def preprocess(inputs, predictor):
    time, lat, lon = inputs
    
    time = pd.to_datetime(time)
    res, var = caliope_model_victor_predict(time, lat, lon)
    road_array = predictor.predict(lat, lon)[0]
    if res is None or isnan(res):
        res = 0
    if var is None or isnan(var):
        var = 0

    new_vars_array = [
        res * var,
        res / (var + 1e-9),
        res - var,
        res**2 + var**2,
        res**2 - var**2,
        res * var**2,
        sum(road_array),                       # Sum of the list
        list_mean(road_array),                 # Mean of the list
        list_std(road_array),                  # Standard deviation of the list
        road_array[0]**2 if len(road_array) > 0 else 0, 
        road_array[1]**2 if len(road_array) > 1 else 0, 
        road_array[2]**2 if len(road_array) > 2 else 0, 
        road_array[3]**2 if len(road_array) > 3 else 0, 
        road_array[4]**2 if len(road_array) > 4 else 0, 
        road_array[5]**2 if len(road_array) > 5 else 0, 
    ]
    combined_array = [res, var] + road_array + new_vars_array
    outputs = torch.tensor(combined_array)
    outputs = outputs.float()
    return outputs


class SimpleModel(nn.Module):
    def __init__(self, input_dim=23, hidden_dim=64, output_dim=1):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    df = pd.read_csv("our_generated_data.csv")
    model = SimpleModel()
    initial_i = 0
    # model.load_state_dict(torch.load(f"pths/model{initial_i-1}.pth"))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    openMapPredictor = OpenTransportKNN()

    # train_df, test_df = train_test_split(df, test_size=0.995, random_state=42)
    # train_df, val_df = train_test_split(train_df, test_size=0.5, random_state=42)
    # y_val_tensor = torch.tensor(val_df['nox_no2'].values, dtype=torch.float32)

    # train_df, test_df = train_test_split(df, test_size=0.7, random_state=42)
    train_df, val_df = train_test_split(df, test_size=0.25, random_state=42)
    y_val_tensor = torch.tensor(val_df['nox_no2'].values, dtype=torch.float32)
    

    # print size of train_df
    print("-"*100)
    print(f"Size of train_df: {train_df.shape[0]}")
    # print(f"Size of test_df: {test_df.shape[0]}")
    print(f"Size of val_df: {val_df.shape[0]}")


    for epoch in range(100):
        # for inputs, target in dataloader:
        init = time.time()
        print(f"------------------------------- EPOCH {epoch} -------------------------------")
        for index, row in train_df.iterrows():
            timee, lat, lon, no2, road_array = row['time'], row['latitude'], row['longitude'], row['nox_no2'], row['indexCarreteres']
            inputs = (timee, lat, lon, road_array)
            inputs = preprocess(inputs, openMapPredictor)
            optimizer.zero_grad()
            outputs = model(inputs)
            target = row['nox_no2'] * CONVERSION_FACTOR
            target = torch.tensor(target, dtype=torch.float32)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
        

        model.eval()
        val_loss = 0.0
        y_trues = []
        y_preds = []
        with torch.no_grad():
            for index, row in val_df.iterrows():
                timee, lat, lon, no2, road_array = row['time'], row['latitude'], row['longitude'], row['nox_no2'], row['indexCarreteres']
                inputs = (timee, lat, lon, road_array)
                inputs = preprocess(inputs, openMapPredictor)
                outputs = model(inputs).unsqueeze(0)
                target = row['nox_no2'] * CONVERSION_FACTOR
                target = torch.tensor(target, dtype=torch.float32)
                val_loss += criterion(outputs, target).item()
                
                y_trues.append(target.item())
                y_preds.append(outputs.squeeze().item())

        val_loss /= len(val_df)
        print(f"Validation Loss: {val_loss:.4f}")
        # Print r-squared
        r2 = r2_score(y_trues, y_preds)
        print(f"R-squared: {r2:.4f}")
            

        torch.save(model.state_dict(), f'pths/model{epoch+initial_i}.pth')
        print(f"Time taken: {time.time() - init}")


    torch.save(model.state_dict(), 'model.pth')