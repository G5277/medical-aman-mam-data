import torch
import pandas as pd
import pickle as pkl
from cnn.class_CNN import model_CNN
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


cnet = model_CNN()
with open('cnn/model.pt', 'rb') as f:
    cnet.load_state_dict(torch.load(f))

data = pd.read_excel("SAMPLE_DATA.xlsx", engine="openpyxl")
X_test = data[["m1", "m2", "m3"]].values
X_test = StandardScaler().fit_transform(X_test)
X_test = torch.tensor(X_test).float().unsqueeze(1)

X_label = torch.tensor(data["Movement"].values)

cnet.eval()
with torch.no_grad():
    pred = cnet(X_test)
    pred_classes = [torch.argmax(output) + 1 for output in pred]
    features_cnn = pred.detach().numpy()  # Convert predictions to NumPy array

    # print("CNN Predictions:", pred_classes)


# FOR RANDOM FOREST
# with open('ml_models/random_forest_model.pkl', 'rb') as f:
#     loaded_rf_model = pkl.load(f)
# predictions_rf = loaded_rf_model.predict(X_test.squeeze().numpy())
# # print("Random Forest Predictions:", predictions_rf)

# combined_features = pd.DataFrame({
#     "cnn_feature": features_cnn[:, 0],  
#     "rf_prediction": predictions_rf
# })

# FOR XGBOOST
with open('ml_models/xgboost_model.pkl', 'rb') as f:
    loaded_xg_model = pkl.load(f)

predictions_xg = loaded_xg_model.predict(X_test.squeeze().numpy())

combined_features = pd.DataFrame({
    "cnn_features" : features_cnn[:,0],
    "xg_prediction" : predictions_xg
})

logistic_model = LogisticRegression()
logistic_model.fit(combined_features, X_label.numpy()) 

# Predict with Logistic Regression
final_predictions = torch.tensor(logistic_model.predict(combined_features)).to('cpu')
# print("Logistic Regression Predictions:", final_predictions)


# Evaluate performance
accuracy = accuracy_score(final_predictions, X_label)
print(f'Accuracy: {accuracy:.2f}') 