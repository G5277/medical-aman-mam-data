from class_CNN import model_CNN
import pandas as pd
import os
import sys
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm

if (os.path.exists("./data/emg_sample.xlsx")):
    data = pd.read_excel("./data/emg_sample.xlsx", engine="openpyxl")
else:
    print("File doesn't exist.")

# print(data.head())
# print(len(data))

# drop rows with null values
data = data.dropna()
# print(len(data))

# print(data)

# separate features and target values
X = data[["m1", "m2", "m3"]].values
y = data["Movement"].values - 1

# Normalize the features
X = StandardScaler().fit_transform(X)
# print(X)

# test train split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Converting to torch tensors
X_train = torch.tensor(X_train).float()
# X_test = torch.tensor(X_test).float()
y_train = torch.tensor(y_train).long()
# y_test = torch.tensor(y_test)#.float()

# Reshaping for CNN fit
X_train = X_train.unsqueeze(1)
# print(X_train[0].shape[1])
# X_test = X_test.unsqueeze(1)
# print(X_test.shape)

cnet = model_CNN()
optimizer = torch.optim.Adam(cnet.parameters(), lr=1e-3)
loss_function = torch.nn.CrossEntropyLoss()

print("start train")


def train_model():
    EPOCHS = 50
    BATCH = 50
    prev = float('inf')
    loss = sys.maxsize + 5
    for epoch in range(EPOCHS):
        if (abs(loss - prev) > 0.001):
            for i in tqdm(range(0, len(X_train), BATCH)):
                prev = loss
                batch_X = X_train[i:i+BATCH]
                # print(f"Size X {len(batch_X)}")
                batch_y = y_train[i:i+BATCH]
                # print(f"Size Y {len(batch_y)}")
                cnet.zero_grad()
                output = cnet(batch_X)
                loss = loss_function(output, batch_y)
                loss.backward()
                optimizer.step()
        else:
            print(f"ending, {loss-prev}")
        print(f"{epoch} : {loss}")

    with open("model.pt", "wb") as f:
        torch.save(cnet.state_dict(), f)


train_model()
