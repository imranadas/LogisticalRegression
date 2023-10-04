import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("data_urban-growth-HW2.csv")
X = data[['Distance to Highways', 'Distance to Major Roads', 'Slope',
          'Distance to CBD', 'Distance to MIDC', 'Distance to Stations',
          'Distance to Hospitals', 'Distance to Airport']]
y = data['Built-up']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)

y_tree_pred = tree_model.predict(X_val)
accuracy_tree = accuracy_score(y_val, y_tree_pred)

print("Decision Tree Model Accuracy:", accuracy_tree)
