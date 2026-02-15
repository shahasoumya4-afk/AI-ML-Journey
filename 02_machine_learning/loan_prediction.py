import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Sample dataset
data = {
    "income": [25000, 40000, 60000, 80000, 120000, 30000, 70000, 90000],
    "age": [22, 25, 35, 45, 52, 23, 40, 60],
    "credit_score": [600, 650, 700, 720, 800, 580, 690, 750],
    "approved": [0, 0, 1, 1, 1, 0, 1, 1]
}

df = pd.DataFrame(data)

# Features and target
X = df[["income", "age", "credit_score"]]
y = df["approved"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))

# Test new applicant
new_person = [[75000, 30, 710]]
result = model.predict(new_person)

if result[0] == 1:
    print("Loan Approved ✅")
else:
    print("Loan Rejected ❌")

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plot_tree(model, feature_names=X.columns, class_names=["Rejected", "Approved"], filled=True)
plt.show()
