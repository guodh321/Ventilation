import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from tensorflow import keras
import tensorflow as tf
import sys

sys.path.append("..")

os.environ["CUDA_VISIBLE_DEVICES"] = f"{3}"

root_path = "/home/dg321/gitTest/PRI/irp/Ventilation/direk"

# Load data
train_1 = pd.read_csv("/home/dg321/gitTest/PRI/irp/Ventilation/direk/data_new/6GIC_meeting_room_sensors_2023-11-09.csv")
print(train_1.shape)
print(train_1.head(5))
print(train_1["time"][0])
print(train_1["time"][158721])

train_1['time'] = pd.to_datetime(train_1['time'])
print(train_1.tail(5))
print(train_1.columns)

column_names_list = train_1.columns.tolist()
train_1.info()

# Plot data over the whole time period
specific_date = pd.to_datetime('2023-10-01')
specific_date1 = pd.to_datetime('2023-10-31')
data_after_specific_date = train_1[(train_1['time'] >= specific_date) & (train_1['time'] <= specific_date1)]
columns_to_select = ["temperature_Main", "Occupancy", "door_gap", "window_gap"] + list(train_1.columns[71:])
data_after_specific_date_selected = data_after_specific_date[columns_to_select]

for i in range(data_after_specific_date_selected.shape[1]):
    plt.figure(figsize=(12, 6))
    plt.plot(data_after_specific_date['time'], data_after_specific_date_selected.iloc[:, i], label=column_names_list[i])
    plt.xlabel('Timestamp')
    plt.ylabel('Your Value')
    plt.title(data_after_specific_date_selected.columns[i])
    plt.legend()
    plt.show()

print(data_after_specific_date_selected.info())

unseen_test_data = data_after_specific_date_selected.iloc[40000:, :]
print(unseen_test_data.shape)
data_after_specific_date_selected = data_after_specific_date_selected.iloc[:40000, :]

# Building Model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

X = data_after_specific_date_selected.drop(columns=["temperature_Main"]).values
y = data_after_specific_date_selected["temperature_Main"].values
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1).to(device)
X_val_tensor = torch.FloatTensor(X_val).to(device)
y_val_tensor = torch.FloatTensor(y_val).view(-1, 1).to(device)

input_size = X_train.shape[1]

class TemperaturePredictionModel(nn.Module):
    def __init__(self, input_size):
        super(TemperaturePredictionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = TemperaturePredictionModel(input_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 1000
batch_size = 32

for epoch in range(num_epochs):
    for i in range(0, len(X_train_tensor), batch_size):
        inputs = X_train_tensor[i:i+batch_size]
        labels = y_train_tensor[i:i+batch_size]
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)

    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), '/home/dg321/gitTest/PRI/irp/Ventilation/23-11/models/temperature_prediction_model_epoch{}.pth'.format(num_epochs))

# Predict on test
X_test = unseen_test_data.drop(columns=["temperature_Main"]).values
y_test = unseen_test_data["temperature_Main"].values

X_test = scaler.transform(X_test)

# Convert val data to PyTorch tensor and move to GPU
X_val_tensor = torch.FloatTensor(X_val).to(device)

# Set the model to evaluation mode
model.eval()

# Make predictions on the val set
with torch.no_grad():
    predictions = model(X_val_tensor).cpu().numpy()

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(y_val[:3000], label='Actual Temperature', linewidth=2)
plt.plot(predictions[:3000], label='Predicted Temperature', linestyle='--', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.title('Actual vs. Predicted Temperature Over Time')
plt.legend()
plt.show()




