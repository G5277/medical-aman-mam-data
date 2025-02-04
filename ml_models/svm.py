import pandas as pd
import pickle as pkl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_excel("emg.xlsx", engine="openpyxl")

# Features and target
X = data[["m1", "m2", "m3"]]
y = data["Movement"]

# test train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# pre-process
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# modelllll
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_train, y_train)

# pred
y_pred = svm_model.predict(X_test)

# eval
print("SVM Accuracy:", accuracy_score(y_test, y_pred))

# save model
with open('ml_models/svm_model.pkl', 'wb') as file:
    pkl.dump(svm_model, file)

# Load the model back
with open('ml_models/svm_model.pkl', 'rb') as file:
    loaded_svm_model = pkl.load(file)

# Make predictions with the loaded model
predictions = loaded_svm_model.predict(X_test)
