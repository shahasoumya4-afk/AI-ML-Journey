# ---------------------------------------------------
# Import Libraries
# ---------------------------------------------------

import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier


# ---------------------------------------------------
# Step 1: Load Dataset
# ---------------------------------------------------

data = load_breast_cancer()

# Features
X = data.data

# Target labels
y = data.target


# ---------------------------------------------------
# Step 2: Train-Test Split
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# ---------------------------------------------------
# Step 3: Train Baseline Model
# ---------------------------------------------------

model = RandomForestClassifier(
    random_state=42,
    n_jobs=-1   # Use all CPU cores for faster training
)

model.fit(X_train, y_train)


# ---------------------------------------------------
# Step 4: Evaluate Baseline Model
# ---------------------------------------------------

baseline_score = model.score(X_test, y_test)

print("Baseline Test Accuracy:", baseline_score)


# ---------------------------------------------------
# Step 5: Apply K-Fold Cross Validation
# ---------------------------------------------------

cv_scores = cross_val_score(model, X_train, y_train, cv=5)

print("Cross Validation Scores:", cv_scores)

print("Average CV Score:", cv_scores.mean())


# ---------------------------------------------------
# Step 6: Define Hyperparameter Search Space
# ---------------------------------------------------

param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10]
}


# ---------------------------------------------------
# Step 7: Randomized Search (Faster Hyperparameter Tuning)
# ---------------------------------------------------

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_distributions=param_grid,
    n_iter=10,        # Test only 10 random combinations
    cv=5,             # 5-fold cross validation
    random_state=42,
    n_jobs=-1
)


# ---------------------------------------------------
# Step 8: Train Random Search
# ---------------------------------------------------

random_search.fit(X_train, y_train)


# ---------------------------------------------------
# Step 9: Best Hyperparameters
# ---------------------------------------------------

print("Best Parameters:", random_search.best_params_)

print("Best Cross Validation Score:", random_search.best_score_)


# ---------------------------------------------------
# Step 10: Evaluate Best Model on Test Data
# ---------------------------------------------------

best_model = random_search.best_estimator_

final_score = best_model.score(X_test, y_test)

print("Final Test Accuracy:", final_score)