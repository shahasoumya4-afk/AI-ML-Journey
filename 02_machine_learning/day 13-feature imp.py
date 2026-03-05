import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

precision_before = precision_score(y_test, y_pred)

print("Precision before removing features:", precision_before)

importances = model.feature_importances_

feature_importance = pd.Series(importances, index=X.columns)

feature_importance = feature_importance.sort_values(ascending=False)

print(feature_importance)

important_features = feature_importance[feature_importance > 0.01].index

X_reduced = X[important_features]

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_reduced, y, test_size=0.2, random_state=42
)

model2 = RandomForestClassifier()

model2.fit(X_train2, y_train2)

y_pred2 = model2.predict(X_test2)

precision_after = precision_score(y_test2, y_pred2)

print("Precision after removing features:", precision_after)

print("Precision before:", precision_before)
print("Precision after:", precision_after)