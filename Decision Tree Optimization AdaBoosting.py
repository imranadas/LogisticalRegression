import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("data_urban-growth-HW2.csv")
X = data[['Distance to Highways', 'Distance to Major Roads', 'Slope',
          'Distance to CBD', 'Distance to MIDC', 'Distance to Stations',
          'Distance to Hospitals', 'Distance to Airport']]
y = data['Built-up']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

weak_learner = DecisionTreeClassifier(max_depth=1)

adaboost_model = AdaBoostClassifier(estimator=weak_learner, n_estimators=100, random_state=42)
adaboost_model.fit(X_train, y_train)

y_adaboost_pred = adaboost_model.predict(X_val)
accuracy_adaboost = accuracy_score(y_val, y_adaboost_pred)

print("AdaBoost Model Accuracy:", accuracy_adaboost)
