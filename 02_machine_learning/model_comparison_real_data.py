import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("loan_data.csv")

# Check target values
print("Loan status values:", df["loan_status"].unique())

# Remove missing values
df = df.dropna()

# Features and target
X = df[["person_income", "loan_amnt", "loan_int_rate", "credit_score"]]
y = df["loan_status"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_acc = accuracy_score(y_test, log_model.predict(X_test))

# Decision Tree
tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)
tree_acc = accuracy_score(y_test, tree_model.predict(X_test))

print("Logistic Regression Accuracy:", log_acc)
print("Decision Tree Accuracy:", tree_acc)
