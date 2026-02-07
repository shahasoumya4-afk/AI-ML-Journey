import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Step 1: Create dataset
data = {
    "hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "score": [30, 40, 50, 60, 70, 80, 85, 95]
}

df = pd.DataFrame(data)

# Step 2: Features and labels
X = df[["hours"]]
y = df["score"]

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Step 4: Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predict
predictions = model.predict(X_test)
print("Predicted scores:", predictions)

# Step 6: Test with new data
new_hours = [[5.5]]
predicted_score = model.predict(new_hours)

print("Predicted score for 5.5 hours:", predicted_score[0])


print("Slope (m):", model.coef_[0])
print("Intercept (b):", model.intercept_)

