import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Create a small dataset
data = {
    "message": [
        "Win money now",
        "Claim your free prize",
        "Meeting at 5pm",
        "Project deadline tomorrow",
        "Free lottery ticket",
        "Call me when you reach",
        "Congratulations you won",
        "Let’s have lunch",
        "Exclusive offer just for you",
        "See you in class"
    ],
    "label": [1,1,0,0,1,0,1,0,1,0]  # 1 = spam, 0 = not spam
}

df = pd.DataFrame(data)

# Step 2: Split features and labels
X = df["message"]
y = df["label"]

# Step 3: Convert text into numbers
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Step 4: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.3, random_state=42
)

# Step 5: Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 6: Predict
predictions = model.predict(X_test)

# Step 7: Evaluation
print("Accuracy:", accuracy_score(y_test, predictions))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))

# Step 8: Test with a new message
new_message = ["Meeting tommorow"]
new_vector = vectorizer.transform(new_message)
result = model.predict(new_vector)

if result[0] == 1:
    print("\nPrediction: Spam")
else:
    print("\nPrediction: Not Spam")
