import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

corr = df.corr()

# Sort correlation with target
target_corr = corr['target'].sort_values(ascending=False)
print("Correlation with target:\n")
print(target_corr)

top_features = target_corr.index[1:4]   # skipping 'target' itself

print("\nSelected Features:\n", list(top_features))

X_all = df.drop('target', axis=1)
X_selected = df[top_features]
y = df['target']

# Train-test split
X_train_all, X_test_all, y_train, y_test = train_test_split(X_all, y, test_size=0.2, random_state=42)
X_train_sel, X_test_sel, _, _ = train_test_split(X_selected, y, test_size=0.2, random_state=42)



scaler_all = StandardScaler()
X_train_all = scaler_all.fit_transform(X_train_all)
X_test_all = scaler_all.transform(X_test_all)

scaler_sel = StandardScaler()
X_train_sel = scaler_sel.fit_transform(X_train_sel)
X_test_sel = scaler_sel.transform(X_test_sel)



model_all = KNeighborsClassifier(n_neighbors=5)
model_all.fit(X_train_all, y_train)

model_sel = KNeighborsClassifier(n_neighbors=5)
model_sel.fit(X_train_sel, y_train)

