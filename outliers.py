import os
import sys
import torch
import numpy as np
import pandas as pd

if os.path.exists("./data/emg.xlsx"):
    data = pd.read_excel("./data/emg.xlsx", engine="openpyxl")
else:
    print("File doesn't exist.")
    sys.exit(1)

data = data.dropna()
# Outlier analysis
Q1 = np.percentile(data['m1'], 25, method='midpoint')
Q3 = np.percentile(data['m1'], 75, method='midpoint')
IQR = Q3 - Q1
print(IQR)
data = data[data['m1'] < (Q3 + 1.5*(IQR))]
data = data[data['m1'] > (Q1 - 1.5*(IQR))]
# print(data)

data = data[data['m2'] < (Q3 + 1.5*(IQR))]
data = data[data['m2'] > (Q1 - 1.5*(IQR))]
# print(data)

data = data[data['m3'] < (Q3 + 1.5*(IQR))]
data = data[data['m3'] > (Q1 - 1.5*(IQR))]
print(data)

data.to_csv('./data/clean.csv', index=False)
