# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas
2. Import Decision tree classifier
3. Fit the data in the model
4. Find the accuracy score


## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: VIGNESH KUMARAN N S
RegisterNumber:  212222230171
*/
```
```python
import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()    #no departments and no left
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```
## Output:
#### data.head():
<img src = "https://github.com/Adhithyaram29D/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393540/2930807a-1e18-460b-9adc-ce2935c6318c" width="600">

#### data.info():
<img src = "https://github.com/Adhithyaram29D/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393540/33512a36-aced-42fe-8a2c-7b7d3cfc31e7" width="200">

#### data.isnull.sum():
<img src = "https://github.com/Adhithyaram29D/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393540/f57954cb-cd4e-4a58-adb0-c054a7fe031e" width="200">

####  df['left'].value_counts():
![image](https://github.com/Adhithyaram29D/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393540/5961c484-d9e4-491f-a314-32a0f4f83615)

#### Label Encoding for String values:
<img src = "https://github.com/Adhithyaram29D/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393540/b1200645-3fda-43ff-8b6f-4761611caf72" width="600">

#### x.head():
<img src = "https://github.com/Adhithyaram29D/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393540/99ba5131-7eaa-4951-b2ff-17fac8b62287" width="600">

#### Accuracy: 
<img src = "https://github.com/Adhithyaram29D/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393540/3e898087-a5d2-4710-9b3b-4f04f336c160" width="200">

#### Prediction:
![image](https://github.com/Adhithyaram29D/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393540/baa8b032-fd61-4a11-8f61-86ab42d1a97a)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
