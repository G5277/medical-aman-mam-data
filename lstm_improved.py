import os
import sys
import torch
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

if os.path.exists("emg.xlsx"):
    data = pd.read_excel("emg.xlsx", engine="openpyxl")
else:
    print("File doesn't exist.")
    sys.exit(1)

data = data.dropna()
X = data[["m1", "m2", "m3"]].values
y = data["Movement"].values - 1  

# X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train).float()
X_test = torch.tensor(X_test).float()
y_train = torch.tensor(y_train).long()
y_test = torch.tensor(y_test).long()

X_train = X_train.unsqueeze(1)
X_test = X_test.unsqueeze(1)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
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
num_epochs = 50
batch_size = 30

lstm_model = LSTMModel(input_size, hidden_size, num_layers, output_size)
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)
loss_function = nn.CrossEntropyLoss()

# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

def train_lstm_model():
    prev_loss = float("inf")
    for epoch in range(num_epochs):
        lstm_model.train()
        total_loss = 0
        for i in tqdm(range(0, len(X_train), batch_size)):
            batch_X = X_train[i:i + batch_size]
            batch_y = y_train[i:i + batch_size]

            optimizer.zero_grad()
            outputs = lstm_model(batch_X)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), max_norm=5) 
            optimizer.step()

        # scheduler.step(total_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

        if abs(prev_loss - loss.item()) < 0.0001:
            print("Loss stabilized. Stopping early.")
            break
        prev_loss = loss

    torch.save(lstm_model.state_dict(), "lstm_improved_model.pt")
    print("Model saved as lstm_improved_model.pt")

# Evaluation function
def evaluate_lstm_model():
    lstm_model.eval()
    with torch.no_grad():
        predictions = lstm_model(X_test)
        predictions = torch.argmax(predictions, dim=1)
        accuracy = (predictions == y_test).sum().item() / len(y_test)
        print(f"Test Accuracy: {accuracy:.2f}")

        # Print confusion matrix and classification report
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, predictions.cpu()))
        print("Classification Report:")
        print(classification_report(y_test, predictions.cpu()))

# Train and evaluate
if __name__ == "__main__":
    TRAIN = True
    if TRAIN:
        train_lstm_model()
    evaluate_lstm_model()