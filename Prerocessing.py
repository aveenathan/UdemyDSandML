#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 23:10:42 2019

@author: Anand
"""
# importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# creating the dataset from a csv file

dataset = pd.read_csv('Data.csv') 


# create the Feature Matrix X and output Vecotr y

X=dataset.iloc[:,:-1].values 
y=dataset.iloc[:,3].values  

# Take care of missing data using a mean of column vector (axis = 0)  replacement imputer

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values=np.nan, strategy="mean", axis=0) 

# Fit the Imputer to the Feature Matrix to columns with missing data
imputer = imputer.fit(X[:,1:3])

# Transform replaces the missing values with the mean of the column values
X[:,1:3]= imputer.transform(X[:,1:3])

# Transform categorical data (textual) into numerical data
# Country values to be encoded

from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])


# Numerical data can unnecessarily make some values more favorable to math model of the algorithm
# Country values to be one hot encoded

from sklearn.preprocessing import OneHotEncoder
onehotencoder_X =OneHotEncoder(categorical_features=[0])
X=onehotencoder_X.fit_transform(X).toarray()

# Dependant variable does not have an ordinal impact, so does not need the dummy variables

labelencoder_y = LabelEncoder()
y=labelencoder_y.fit_transform(y)

# Splitting the data set into a training and test data set
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=.20,random_state=0 )

# Feature scaling - done to preserve the Euclidean distances, meaning, the actual values of the variable
# Influence more - so Age is usually under 100 and Salaries are in 1000's Euclidean distance
#is impacted by the values themselves. 
# So we scale them to the same by either standardization 
# Observation - mean(feature) / Standard deviation 
# Or we scale them using normalization
# Observation - min(feature) / max(feature) - min(feature)
# This example uses StandardScaler

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)



