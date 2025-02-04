import pandas as pd
import pickle as pkl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# Load data
data = pd.read_excel("./data/emg_sample.xlsx", engine="openpyxl")

# Features and target
X = data[["m1", "m2", "m3"]]
y = data["Movement"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train XGBoost Classifier
xgb_model = XGBClassifier(n_estimators=50, learning_rate=0.1, random_state=5)
xgb_model.fit(X_train, y_train - 1)

# Make predictions
y_pred = xgb_model.predict(X_test)

# Evaluate the model
print("XGBoost Accuracy:", accuracy_score(y_test-1, y_pred))

# XG-BOOST MODEL
with open('xgboost_model.pkl', 'wb') as file:
    pkl.dump(xgb_model, file)

# Load the model back
with open('xgboost_model.pkl', 'rb') as file:
    loaded_xgb_model = pkl.load(file)

# Make predictions with the loaded model
predictions = loaded_xgb_model.predict(X_test)
