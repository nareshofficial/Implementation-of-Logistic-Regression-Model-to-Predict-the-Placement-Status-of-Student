# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

Step-1 : Import the required packages and print the present data.
Step-2 : Print the placement data and salary data.
Step-3 : Find the null and duplicate values. 
Step-4 : Using logistic regression find the predicted values of accuracy , confusion matrices.
Step-5 : Display the results.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: NARESH.P.S
RegisterNumber: 212223040127
*/
import pandas as pd
data=pd.read_csv("/content/Placement_Data (1).csv")
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1.isnull()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["spefcialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear") 
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:

![image](https://github.com/Bosevennila/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144870486/2a5ea6b4-66a3-43d9-8230-2fa6f1ea1189)

![image](https://github.com/Bosevennila/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144870486/3a5132f8-a452-458c-8a35-babb20ceeeba)

![image](https://github.com/Bosevennila/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144870486/887bf772-e251-4380-a82c-aa935ca5df7c)

![image](https://github.com/Bosevennila/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144870486/654889ce-eede-4b48-9c6d-634668012680)

![image](https://github.com/Bosevennila/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144870486/0d464901-49ab-456e-b083-953acb3d1113)

![image](https://github.com/Bosevennila/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144870486/9df90624-b735-46c5-adc8-78c1ffb834bc)

![image](https://github.com/Bosevennila/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144870486/db691656-9a9e-4c4b-a50e-c3686137ddfc)

![image](https://github.com/Bosevennila/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144870486/17e0f696-b95d-4607-bc40-56c0176aab1e)

![image](https://github.com/Bosevennila/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/144870486/24050d7c-0866-4ac9-bf47-68cbcd786205)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
