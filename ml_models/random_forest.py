import pandas as pd
import pickle as pkl
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# data
data = pd.read_excel("./data/emg_sample.xlsx", engine="openpyxl")

# Features and target
X = data[["m1", "m2", "m3"]]
y = data["Movement"]

# test train split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# pre-processing
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# test
y_pred = rf_model.predict(X_test)

# eval
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))

# Save the model
# with open('ml_models/random_forest_model.pkl', 'wb') as file:
#     pkl.dump(rf_model, file)

# # Load the model back
# with open('ml_models/random_forest_model.pkl', 'rb') as file:
#     loaded_rf_model = pkl.load(file)

# Make predictions with the loaded model
# predictions = loaded_rf_model.predict(X_test)
