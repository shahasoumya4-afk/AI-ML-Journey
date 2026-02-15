import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# -------------------------
# Step 1: Create dataset
# -------------------------
data = {
    "age": [25, 45, 35, 50, 23, 40, 60, 48, 33, 55],
    "income": [20000, 50000, 30000, 80000, 18000, 45000, 90000, 60000, 32000, 75000],
    "credit_score": [400, 700, 650, 800, 350, 690, 820, 720, 600, 780],
    "approved": [0, 1, 1, 1, 0, 1, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

X = df[["age", "income", "credit_score"]]
y = df["approved"]

# -------------------------
# Step 2: Train model
# -------------------------
model = DecisionTreeClassifier()
model.fit(X, y)

# -------------------------
# Step 3: Streamlit UI
# -------------------------
st.title("🏦 Loan Approval Predictor")
st.markdown("### 💡 This app predicts whether a loan will be approved based on:")
st.markdown("- Age")
st.markdown("- Income")
st.markdown("- Credit Score")

st.write("Enter applicant details:")

age = st.slider("Age", 18, 65, 30)
income = st.number_input("Income", 10000, 100000, 30000)
credit = st.slider("Credit Score", 300, 850, 650)

# -------------------------
# Step 4: Prediction
# -------------------------
if st.button("Check Loan Approval"):
    input_data = [[age, income, credit]]
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Loan Approved ✅")
    else:
        st.error("Loan Rejected ❌")
st.markdown("---")
st.caption("Built with Decision Tree Classifier | Day 7 ML Project")
