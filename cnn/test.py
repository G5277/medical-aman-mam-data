import torch
from torch import load
from class_CNN import model_CNN
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("./data/final_data.csv")  # , engine="openpyxl")
X = data[["m1", "m2", "m3"]].values
X = StandardScaler().fit_transform(X)
X = torch.tensor(X)
X = X.float().unsqueeze(1)
y = torch.tensor(data["Movement"].values)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


cnet = model_CNN()

with open("model.pt", "rb") as f:
    cnet.load_state_dict(load(f), strict=True)

win = 0
lose = 0
for i in range(0, len(X_test)):
    batch_y = X_test[i:i+1]
    # print(batch_y)
    output = cnet(batch_y)
    if (torch.argmax(output) == y_test[i]-1):
        win = win + 1
    else:
        print(torch.argmax(output), y_test[i]-1)
        lose = lose + 1


print(f"Accuracy {win/(win+lose)}")
print(f"{lose}")
