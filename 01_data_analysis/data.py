import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("students.csv")   # or iris.csv

# Show first rows
print(data.head())

# Dataset info
print(data.info())

# Basic statistics
print(data.describe())

# Average sepal length
avg_sepal = data["sepal_length"].mean()
print("Average Sepal Length:", avg_sepal)

# Count species
print(data["species"].value_counts())

# Plot
plt.hist(data["sepal_length"])
plt.title("Distribution of Sepal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Frequency")
plt.show()

# Insights:
# 1. The dataset contains three species of flowers.
# 2. Average sepal length is around ___.
# 3. Most values fall between ___ and ___.
