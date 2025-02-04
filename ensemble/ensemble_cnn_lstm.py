import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from lstm.lstm import LSTMModel
import torch.optim as optim
from cnn.class_CNN import model_CNN
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

if os.path.exists("emg.xlsx"):
    data = pd.read_excel("emg.xlsx", engine="openpyxl")
else:
    print("File doesn't exist.")
    sys.exit(1)

data = data.dropna()
X = data[["m1", "m2", "m3"]].values
y = data["Movement"].values - 1  
X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train).float().unsqueeze(1)  
X_test = torch.tensor(X_test).float().unsqueeze(1)
y_train = torch.tensor(y_train).long()
y_test = torch.tensor(y_test).long()

cnet = model_CNN()
with open("cnn/model.pt", "rb") as f:
    cnet.load_state_dict(torch.load(f), strict=True)

with torch.no_grad():
    preds_cnn_train = torch.softmax(cnet(X_train), dim=1).detach().numpy()
    preds_cnn_test = torch.softmax(cnet(X_test), dim=1).detach().numpy()

input_size = 3  
hidden_size = 128
num_layers = 2
output_size = len(set(y)) 

lstm_model = LSTMModel(input_size, hidden_size, num_layers, output_size)
with open("lstm_model.pt", "rb") as f:
    lstm_model.load_state_dict(torch.load(f, map_location=torch.device("cpu")))

with torch.no_grad():
    preds_lstm_train = torch.softmax(lstm_model(X_train), dim=1).detach().numpy()
    preds_lstm_test = torch.softmax(lstm_model(X_test), dim=1).detach().numpy()

train_meta_features = np.hstack((preds_lstm_train, preds_cnn_train))
test_meta_features = np.hstack((preds_lstm_test, preds_cnn_test))

meta_model = LogisticRegression()
meta_model.fit(train_meta_features, y_train)

final_preds = meta_model.predict(test_meta_features)

device = 'cpu'
y_test = torch.tensor(y_test).to(device)
final_preds = torch.tensor(final_preds).to(device)

# Evaluate performance
accuracy = accuracy_score(y_test, final_preds)
print("Ensemble Accuracy:", accuracy)