# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and load the student dataset.

2. Split the data into training and testing sets.

3. Train the Logistic Regression model using the training data.

4. Predict placement status and evaluate the model accuracy.
 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: KISHORE KUMAR B
RegisterNumber:  212225240073
*/
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv('/content/drive/MyDrive/Placement_Data.csv')
df

df.isnull().sum()

le  = LabelEncoder()
for column in df.columns:
  if df[column].dtype == 'object' :
    df[column] = le.fit_transform(df[column])

X = df.drop(['sl_no', 'status', 'salary'], axis=1)
y = df['status']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:", confusion_matrix(y_test, y_pred))

print("\nClassfication Report:",classification_report(y_test, y_pred))

new_student = [[1,85.5, 0, 80.2, 0, 2, 78.6, 2, 1, 75.0, 0, 70.5]]
new_student = scaler.transform(new_student)
prediction = model.predict(new_student)


if prediction[0] == 0:
    print("Student is not placed")
else:
    print("Student is placed")
```

## Output:
<img width="1920" height="1080" alt="Screenshot 2026-08-18 100840" src="https://github.com/user-attachments/assets/3d6b5d2f-a442-471e-ae05-a9ba803e79d4" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
