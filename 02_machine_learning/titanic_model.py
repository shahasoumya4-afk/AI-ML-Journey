import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

#step 1: load dataset
df = sns.load_dataset("titanic")

#2:select important columns
df=df[["survived","pclass","sex","age","fare"]]

#3:handle missing values
df=df.dropna()

#4:convert gender to numbers
df["sex"] = df["sex"].map({"male": 0, "female":1})

print(df.head())

#5:Features and labels
x=df[["pclass","sex","age","fare"]]
y=df["survived"]

#6: split data
x_train,x_test,y_train,y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

#step 7: train model 
model = LogisticRegression(max_iter=200)
model.fit(x_train, y_train)

#8:predict
predictions = model.predict(x_test)

#9:Accuracy
accuracy = accuracy_score(y_test, predictions)
print("Model accuracy:",accuracy)


#Confussion matrix 
cm= confusion_matrix(y_test, predictions)
print ("Confusion Matrix:\n", cm)

#Detailed report 
print("\n Classification Report:\n")
print(classification_report(y_test, predictions))
    

new_passenger=[[1,1,25,80]]
prediction=model.predict(new_passenger)
if prediction[0] == 1:
    print("Prediction: Survived")
else:
    print("Prediction: Did not survive")



