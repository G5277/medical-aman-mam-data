import os
import sys
import torch
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if os.path.exists("./data/emg_sample.xlsx"):
    data = pd.read_excel("./data/emg_sample.xlsx", engine="openpyxl")
else:
    print("File doesn't exist.")
    sys.exit(1)

data = data.dropna()

X = data[["m1", "m2", "m3"]].values
y = data["Movement"].values - 1

X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train).float()
X_test = torch.tensor(X_test).float()
y_train = torch.tensor(y_train).long()
y_test = torch.tensor(y_test).long()

X_train = X_train.unsqueeze(1)
X_test = X_test.unsqueeze(1)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        # *2 for bidirectional
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return nn.functional.softmax(out, dim=1)


input_size = 3
hidden_size = 128
num_layers = 2
output_size = len(set(y))
learning_rate = 1e-3
num_epochs = 15
batch_size = 40

lstm_model = LSTMModel(input_size, hidden_size, num_layers, output_size)
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)
loss_function = nn.CrossEntropyLoss()

TRAIN = True
if (TRAIN == True):
    def train_lstm_model():
        prev_loss = float("inf")
        for epoch in range(num_epochs):
            for i in tqdm(range(0, len(X_train), batch_size)):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]

                lstm_model.zero_grad()
                outputs = lstm_model(batch_X)
                loss = loss_function(outputs, batch_y)
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

            # if abs(prev_loss - loss.item()) < 0.0000001:
            # print("Loss stabilized. Stopping early.")
            # break
            # prev_loss = loss.item()

        torch.save(lstm_model.state_dict(), "lstm_model.pt")
        print("Model saved as lstm_model.pt")

    train_lstm_model()

    def evaluate_lstm_model():
        lstm_model.eval()
        with torch.no_grad():
            predictions = lstm_model(X_test)
            predictions = torch.argmax(predictions, dim=1)
            accuracy = (predictions == y_test).sum().item() / len(y_test)
            print(f"Test Accuracy: {accuracy:.2f}")

    evaluate_lstm_model()
