import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

sns.countplot(x='target', data=df)
plt.title("Target Distribution")
plt.show()

df['mean radius'].hist()
plt.title("Mean Radius Distribution")
plt.show()

sns.boxplot(x=df['mean area'])
plt.title("Boxplot of Mean Area")
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title("Feature Correlation")
plt.show()

sns.pairplot(df[['mean radius', 'mean texture', 'mean area', 'target']], hue='target')
plt.show()


