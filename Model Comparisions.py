import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("data_urban-growth-HW2.csv")
X = data[['Distance to Highways', 'Distance to Major Roads', 'Slope',
          'Distance to CBD', 'Distance to MIDC', 'Distance to Stations',
          'Distance to Hospitals', 'Distance to Airport']]
y = data['Built-up']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

selected_features_model1 = ['Distance to Highways', 'Slope', 'Distance to CBD']
X_train_model1 = X_train[selected_features_model1]
X_val_model1 = X_val[selected_features_model1]

model1 = LogisticRegression()
model1.fit(X_train_model1, y_train)
y_pred1 = model1.predict(X_val_model1)
accuracy1 = accuracy_score(y_val, y_pred1)

selected_features_model2 = ['Distance to Major Roads', 'Slope', 'Distance to MIDC']
X_train_model2 = X_train[selected_features_model2]
X_val_model2 = X_val[selected_features_model2]

model2 = LogisticRegression()
model2.fit(X_train_model2, y_train)
y_pred2 = model2.predict(X_val_model2)
accuracy2 = accuracy_score(y_val, y_pred2)

adaboost_model = AdaBoostClassifier(n_estimators=50, random_state=42)
adaboost_model.fit(X_train, y_train)
y_adaboost_pred = adaboost_model.predict(X_val)
accuracy_adaboost = accuracy_score(y_val, y_adaboost_pred)

random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, y_train)
y_rf_pred = random_forest_model.predict(X_val)
accuracy_rf = accuracy_score(y_val, y_rf_pred)

models = ['Logistic Regression Model 1', 'Logistic Regression Model 2', 'AdaBoost Model', 'Random Forest Model']
accuracies = [accuracy1, accuracy2, accuracy_adaboost, accuracy_rf]

for model, accuracy in zip(models, accuracies):
    print(f"{model} Accuracy:", accuracy)

improvement_model2 = ((accuracy2 - accuracy1) / accuracy1) * 100
improvement_adaboost = ((accuracy_adaboost - accuracy1) / accuracy1) * 100
improvement_rf = ((accuracy_rf - accuracy1) / accuracy1) * 100

print(f"Improvement of Logistic Regression Model 2 over Model 1: {improvement_model2:.2f}%")
print(f"Improvement of AdaBoost Model over Model 1: {improvement_adaboost:.2f}%")
print(f"Improvement of Random Forest Model over Model 1: {improvement_rf:.2f}%")

