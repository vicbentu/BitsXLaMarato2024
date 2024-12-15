import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from models import caliope_model_victor_predict, OpenTransportKNN
from neural_network import SimpleModel, preprocess, CONVERSION_FACTOR

df = pd.read_csv("updated_data.csv")
# read the model from  "model.pth"
model = SimpleModel()  # Instantiate the model
model.load_state_dict(torch.load("pths/model19.pth"))
model.eval()
openMapPredictor = OpenTransportKNN()

train_df, test_df = train_test_split(df, test_size=0.2)

# take 100 random on test_df
test_df = test_df.sample(100)

X_test = test_df.drop(columns=["nox_no2"])  # Replace "target" with the actual target column name
y_test = test_df["nox_no2"]  # Replace "target" with the actual target column name
criterion = nn.MSELoss()

model.eval()
X_test_tensor = X_test.apply(lambda row: preprocess(row, openMapPredictor), axis=1)
# torch.save(X_test_tensor.tolist(), 'X_test_tensor.pth')


# X_test_tensor = torch.load('X_test_tensor.pth')
X_test_tensor = torch.stack([torch.tensor(item, dtype=torch.float32) for item in X_test_tensor])

y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

print("STARTING INFERENCE")
with torch.no_grad():
    outputs = model(X_test_tensor)
    loss = criterion(outputs.squeeze(), y_test_tensor*CONVERSION_FACTOR)
    print(f'Test Loss: {loss.item()}')
    r2 = r2_score(y_test_tensor.numpy() * CONVERSION_FACTOR, outputs.squeeze().numpy())
    print(f'R-squared: {r2}')
