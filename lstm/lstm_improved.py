import os
import sys
import torch
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

TRAIN = True

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

cpu = torch.device('cpu')

print(device)
if os.path.exists("./data/final_data.csv"):
    data = pd.read_csv("./data/final_data.csv")
else:
    print("File doesn't exist.")
    sys.exit(1)

# if os.path.exists("./data/emg_sample.xlsx"):
#     data = pd.read_excel("./data/emg_sample.xlsx", engine="openpyxl")
# else:
#     print("File doesn't exist.")
#     sys.exit(1)


data = data.dropna()
X = data[["m1", "m2", "m3"]].values
y = data["Movement"].values - 1

# X = MinMaxScaler().fit_transform(X)

# X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train).float()
X_test = torch.tensor(X_test).float()
y_train = torch.tensor(y_train).long()
y_test = torch.tensor(y_test).long()

X_train = X_train.unsqueeze(1).to(device)
X_test = X_test.unsqueeze(1).to(device)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


input_size = 3
hidden_size = 128
num_layers = 2
output_size = len(set(y))
learning_rate = 1e-3
num_epochs = 20
batch_size = 20

lstm_model = LSTMModel(input_size, hidden_size,
                       num_layers, output_size).to(device)
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)
loss_function = nn.CrossEntropyLoss()


def train_lstm_model():
    prev_loss = float("inf")
    for epoch in range(num_epochs):
        lstm_model.train()
        for i in tqdm(range(0, len(X_train), batch_size)):
            batch_X = X_train[i:i + batch_size].to(device)
            batch_y = y_train[i:i + batch_size].to(device)

            optimizer.zero_grad()
            outputs = lstm_model(batch_X)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

        if abs(prev_loss - loss.item()) < 0.0001:
            print("Loss stabilized. Stopping early.")
            # break
        prev_loss = loss.item()

    torch.save(lstm_model.state_dict(), "lstm_improved_model.pt")
    print("Model saved as lstm_improved_model.pt")


# Evaluation function
with open("lstm_improved_model.pt", "rb") as f:
    lstm_model.load_state_dict(torch.load(f), strict=True)


def evaluate_lstm_model():
    lstm_model.eval()
    with torch.no_grad():
        predictions = lstm_model(X_test)
        predictions = torch.argmax(predictions, dim=1).cpu()
        accuracy = (predictions == y_test).sum().item() / len(y_test)
        print(f"Test Accuracy: {accuracy:.2f}")

        # Print confusion matrix and classification report
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, predictions))
        print("Classification Report:")
        print(classification_report(y_test, predictions))
        print(predictions.sum(), len(predictions))

    # Train and evaluates
# TRAIN = False
if TRAIN:
    train_lstm_model()
evaluate_lstm_model()
