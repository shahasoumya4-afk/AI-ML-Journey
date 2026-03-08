# AI-ML-Journey
My 6-month journey to become an AI/ML Engineer.

I am a 2nd-year engineering student building projects daily to gain strong practical skills in machine learning and AI systems.

---

## Learning Roadmap

### Day 1
- Python basics
- Variables, lists, loops, functions
- Mini student analysis project

### Day 2
- Introduction to Machine Learning
- Linear Regression model
- Predicted student scores

### Day 3
- Logistic Regression
- First classification model
- Pass/fail prediction

### Day 4
- Titanic survival prediction
- Data cleaning
- Logistic regression
- ~75% accuracy

### Day 5
- Spam detection using Logistic Regression
- Text preprocessing with CountVectorizer
- Learned precision, recall, F1-score
- Confusion matrix analysis
- Tested custom messages
- Understood how text converts into numerical vectors

### Day 6
- Linear Regression model
- House price prediction
- Features: area, bedrooms, bathrooms
- Learned coefficients and intercept
- Evaluated using MAE and MSE
- Added user input for real-time prediction

### Day 7
- Decision Tree classifier
- Loan approval prediction
- Learned gini, samples, value, class
- Visualized decision tree
- Built Streamlit web app
- Real-time loan approval prediction

### Day 8
- Used real-world loan dataset (CSV)
- Data cleaning (handled missing values)
- Selected important numeric features
- Compared Logistic Regression and Decision Tree
- Achieved ~81% accuracy
- Learned about model evaluation and tuning basics

### Day 9
-Implemented K-Nearest Neighbors (KNN) algorithm
-Understood distance-based learning (Euclidean distance)
-Learned how KNN makes predictions using nearest neighbors
-Explored effect of different K values
-Understood overfitting (K=1) and underfitting (large K)
-Built a basic classification model using KNN
-Compared KNN with previous models

### Day 10
-Learned feature scaling using StandardScaler
-Understood why scaling is important for distance-based algorithms
-Applied scaling on dataset before training KNN
-Visualized KNN decision boundary (2D intuition)
-Understood how KNN forms non-linear decision regions
-Explored impact of K on decision boundary shape
-Improved model performance using proper preprocessing

### Day 11
-Learned Data Visualization for Exploratory Data Analysis (EDA)
-Used Matplotlib and Seaborn for plotting
-Created Histogram to understand feature distribution
-Used Boxplot to detect outliers
-Used Countplot to analyze class distribution
-Generated Heatmap to study feature correlations
-Created Pairplot to visualize relationships between features
-Gained insights into data patterns, balance, and feature importance[count plot](image.png)
[histogram](image-1.png)
[boxplot](image-2.png)
[heatmap](image-3.png)
[pairplot](image-4.png)

### Day 12
-Explored dataset relationships using correlation analysis
-Generated correlation matrix to measure relationships between features
-Identified features strongly related to the target variable
-Sorted correlations to find most influential features
-Visualized feature relationships for better understanding
-Learned how correlation helps in feature selection and data analysis

### Day 13
-Learned the importance of feature selection in machine learning
-Studied different feature selection methods (Filter, Wrapper, Embedded)
-Implemented Random Forest Classifier
-Extracted feature importance scores from the trained model
-Identified and removed less important features
-Compared model precision before and after feature selection
-Visualized feature importance using graphs

# Day 14

Learned the concepts of overfitting and underfitting

Understood how model complexity affects performance

Studied the bias–variance tradeoff

Learned how high bias leads to underfitting

Learned how high variance leads to overfitting

Explored methods to reduce overfitting (regularization, pruning, cross validation)

Understood the importance of balancing bias and variance for better model generalization

# Day 15

Learned the concept of K-Fold Cross Validation

Understood the limitations of a single train–test split

Studied how datasets are divided into K equal folds

Trained the model multiple times using different folds

Calculated performance scores for each fold

Computed the average cross validation score

Implemented K-Fold Cross Validation using cross_val_score() in sklearn

# Day 16

Learned the concept of hyperparameters in machine learning

Studied important Random Forest hyperparameters (n_estimators, max_depth, min_samples_split)

Built a baseline Random Forest model

Applied cross validation to evaluate model stability

Implemented GridSearchCV to test different hyperparameter combinations

Used RandomizedSearchCV for faster hyperparameter tuning

Selected the best performing model based on cross validation score

Evaluated the final tuned model on the test dataset