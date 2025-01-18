import torch
import pandas as pd
import pickle as pkl
from xgboost import XGBClassifier
from cnn.class_CNN import model_CNN
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

device = 'cpu'

with open('ml_models/random_forest_model.pkl', 'rb') as f:
    loaded_rf_model = pkl.load(f)

cnet = model_CNN()
with open('cnn/model.pt', 'rb') as f:
    cnet.load_state_dict(torch.load(f))

data = pd.read_excel("SAMPLE_DATA.xlsx", engine="openpyxl")
X = data[["m1", "m2", "m3"]].values
X = StandardScaler().fit_transform(X)
X = torch.tensor(X).float().unsqueeze(1)

y = torch.tensor(data["Movement"].values).cpu() - 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)


cnet.eval()
with torch.no_grad():
    pred = cnet(X_train)
    pred_classes = [torch.argmax(output) + 1 for output in pred]
    features_cnn = pred.detach().numpy()  

    # print("CNN Predictions:", pred_classes)

predictions_rf = loaded_rf_model.predict(X_train.squeeze().numpy())
# print("Random Forest Predictions:", predictions_rf)

combined_features = pd.DataFrame({
    "cnn_feature": features_cnn[:, 0],  
    "rf_prediction": predictions_rf
})

combined_test = pd.DataFrame({"cnn_feature" : (cnet(X_test).detach().numpy())[:,0], "rf_prediction" : loaded_rf_model.predict(X_test.squeeze().numpy())})

xgb_model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)

xgb_model.fit(combined_features, y_train)

y_pred = xgb_model.predict(combined_test)

a = torch.tensor(y_test).to(device)
b = torch.tensor(y_pred).to(device)

# Evaluate performance
accuracy = accuracy_score(a.numpy(),b.numpy())
print(f'Accuracy: {accuracy:.2f}')