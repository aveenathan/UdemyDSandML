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
# Splitting the data set into a training and test data set
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=.20,random_state=0 )
# Feature scaling - 
# This is opionally used, so it is Multiline commented
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)                    """



