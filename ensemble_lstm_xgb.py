import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import pickle as pkl
import torch.optim as optim
from lstm.lstm_improved import LSTMModel
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier,  RandomForestClassifier

# if os.path.exists("./data/emg_sample.xlsx"):
#     data = pd.read_excel("./data/emg_sample.xlsx", engine="openpyxl")
# else:
#     print("File doesn't exist.")
#     sys.exit(1)

data = pd.read_csv('./data/clean.csv')
data = data.dropna()
X = data[["m1", "m2", "m3"]].values
y = data["Movement"].values - 1
# X = StandardScaler().fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

xgb_model = XGBClassifier(n_estimators=100, max_depth=6,
                          learning_rate=0.1, random_state=42)
# Load the model
with open('./ml_models/xgboost_model.pkl', 'rb') as file:
    loaded_xgb_model = pkl.load(file)

# Make predictions with the loaded model
preds_xgb_train = loaded_xgb_model.predict(X_train)
preds_xgb_test = loaded_xgb_model.predict(X_test)

X_train = torch.tensor(X_train).float().unsqueeze(1)
X_test = torch.tensor(X_test).float().unsqueeze(1)
y_train = torch.tensor(y_train).long()
# y_test = torch.tensor(y_test).long()

input_size = 3
hidden_size = 128
num_layers = 2
output_size = len(set(y))

lstm_model = LSTMModel(input_size, hidden_size, num_layers, output_size)
with open("./lstm/lstm_model.pt", "rb") as f:
    lstm_model.load_state_dict(torch.load(f, map_location=torch.device("cpu")))

with torch.no_grad():
    preds_lstm_train = torch.argmax(
        lstm_model(X_train), dim=1).detach().numpy()
    preds_lstm_test = torch.argmax(lstm_model(X_test), dim=1).detach().numpy()

print(preds_lstm_train[0])
print(preds_xgb_train[0])

train_meta_features = np.column_stack((preds_lstm_train, preds_xgb_train))
test_meta_features = np.column_stack((preds_lstm_test, preds_xgb_test))

print(train_meta_features.shape)
print(y_train.shape)
meta_model = RandomForestClassifier()
meta_model.fit(train_meta_features, y_train)

final_preds = meta_model.predict(test_meta_features)

# y_test = torch.tensor(y_test)
# final_preds = torch.tensor(final_preds)

# Evaluate performance
accuracy = accuracy_score(y_test, final_preds)
print("Ensemble Accuracy:", accuracy)
