# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling skiped as the Regression Model will do it
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train) """

# Simple Linear Regression Model
# y = b0 + b1X
# b1 magnifies/deflates a unit change in x1 impacts unit change in y
# b0, Salary value when Experience is 0 
# b1 - Slope of the line, impact of experience on salary



#Fitting Simple Linear Regression Model to the Taining Set - Ordinary least squares Linear Regression.

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)


# Predcting the test set observations
y_pred=regressor.predict(X_test)

# Plotting the prediction and real observation points

plt.scatter(X_train,y_train,color='red')
plt.plot(X_train, regressor.predict(X_train) , color='blue')
plt.title("Salary vs. Experience (training set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")

# Plotting the prediction for test set data
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train, regressor.predict(X_train) , color='blue')
plt.title("Salary vs. Experience (test set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")

