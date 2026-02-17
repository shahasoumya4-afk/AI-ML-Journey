import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Step 1: Load dataset
df = pd.read_csv("loan_data.csv")

# Step 2: Clean data
df = df.dropna()

# Step 3: Convert target column
df["loan_status"] = df["loan_status"].map({1: 1, 0: 0})

# Step 4: Select numeric features
features = [
    "person_age",
    "person_income",
    "loan_amnt",
    "loan_int_rate",
    "credit_score"
]

X = df[features]
y = df["loan_status"]

# Step 5: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Model 1: Without Scaling
# -------------------------------
model_no_scaling = LogisticRegression(max_iter=1000)
model_no_scaling.fit(X_train, y_train)

pred_no_scaling = model_no_scaling.predict(X_test)
acc_no_scaling = accuracy_score(y_test, pred_no_scaling)

print("Accuracy without scaling:", acc_no_scaling)

# -------------------------------
# Model 2: With StandardScaler
# -------------------------------
scaler = StandardScaler()

# Fit on training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform test data
X_test_scaled = scaler.transform(X_test)

model_scaled = LogisticRegression(max_iter=1000)
model_scaled.fit(X_train_scaled, y_train)

pred_scaled = model_scaled.predict(X_test_scaled)
acc_scaled = accuracy_score(y_test, pred_scaled)

print("Accuracy with scaling:", acc_scaled)
