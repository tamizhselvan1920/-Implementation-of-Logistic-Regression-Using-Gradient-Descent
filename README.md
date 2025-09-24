# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.

2.Load the dataset.

3.Define X and Y array.

4.Define a function for costFunction,cost and gradient.

5.Define a function to plot the decision boundary.

## Program:
```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by:THAMIZH SELVAN R
RegisterNumber:  212222230158

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv('Placement_Data.csv')
dataset
dataset= dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary',axis=1)
dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes
dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
Y
theta=np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss(theta, X, y ):
    h = sigmoid(X.dot(theta)) 
    return -np.sum(y *np.log(h)+ (1- y) *np.log(1-h))
def gradient_descent(theta, x, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h=sigmoid(X.dot(theta))
        gradient=X.T.dot (h-y) /m
        theta-=alpha * gradient
    return theta
theta= gradient_descent (theta,X,y,alpha=0.01, num_iterations=1000)
def predict(theta, X):
    h=sigmoid(X.dot(theta))
    y_pred=np.where( h >= 0.5,1 , 0)
    return y_pred

y_pred= predict(theta,X)
accuracy = np.mean(y_pred.flatten()==y) 
print("Accuracy:", accuracy)
print(Y)
print(y_pred)
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```
## Output:
## DATASET:
<img width="1023" height="387" alt="image" src="https://github.com/user-attachments/assets/b2e4ba65-e47d-4035-8378-84ff7519e578" />

## Labelling data:

<img width="393" height="381" alt="image" src="https://github.com/user-attachments/assets/a5306408-fb5b-476e-8814-d2c2eee7c681" />

## Lablling the column:

<img width="1049" height="479" alt="image" src="https://github.com/user-attachments/assets/74d0695d-838d-4934-9733-15a2c4d2ae02" />

## DependentVariables:

<img width="1072" height="301" alt="image" src="https://github.com/user-attachments/assets/f2c744a2-6273-4141-b339-8b28938e6b3f" />


## Accuracy:

<img width="379" height="60" alt="image" src="https://github.com/user-attachments/assets/6c18c165-b479-425e-9111-500e6620ee8c" />

## Y:

<img width="912" height="184" alt="image" src="https://github.com/user-attachments/assets/34cd306b-2dd8-40dc-8125-7fa4b4ffbe14" />

## New Predicted data:

<img width="695" height="343" alt="image" src="https://github.com/user-attachments/assets/207cbfbf-0f42-4467-ab74-8f52f70f18e0" />



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

