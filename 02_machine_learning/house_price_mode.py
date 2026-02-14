import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Step 1: Create dataset
data = {
    "area": [800, 1000, 1200, 1500, 1800, 2000, 2200, 2500],
    "bedrooms": [2, 2, 3, 3, 3, 4, 4, 5],
    "bathrooms": [1, 2, 2, 2, 3, 3, 3, 4],
    "price": [2000000, 2500000, 3000000, 3800000, 4500000, 5000000, 5500000, 6500000]
}

df = pd.DataFrame(data)

# Step 2: Separate features and target
X = df[["area", "bedrooms", "bathrooms"]]
y = df["price"]

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Step 4: Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predict
predictions = model.predict(X_test)

# Step 6: Evaluate
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)

# Step 7: Test new house
new_house = pd.DataFrame({
    "area": [1400],
    "bedrooms": [3],
    "bathrooms": [2]
})

predicted_price = model.predict(new_house)


print("Predicted Price:", predicted_price[0])
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)


# Take user input
area = float(input("Enter area (sq ft): "))
bedrooms = int(input("Enter number of bedrooms: "))
bathrooms = int(input("Enter number of bathrooms: "))

# Make prediction
new_house = [[area, bedrooms, bathrooms]]
predicted_price = model.predict(new_house)

print("Predicted House Price:", predicted_price[0])
