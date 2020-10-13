#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 23:04:21 2020

@author: suryanshsoni
"""

#data preprocessing 

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import keras



dataset = pd.read_csv('Churn_modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_1.fit_transform(X[:,2])


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X=np.array(columnTransformer.fit_transform(X),dtype=np.str)

X = X[:,1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#build neural net
from keras.models import Sequential
from keras.layers import Dense
classifier = Sequential() 

#adding first layer 
classifier.add(Dense(activation='relu',units = 6,kernel_initializer="uniform",input_shape = (11,)))
classifier.add(Dense(activation='relu',units = 6,kernel_initializer="uniform"))
classifier.add(Dense(activation='sigmoid',units = 1,kernel_initializer="uniform"))

#compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics = ['accuracy'])
classifier.fit(X_train,y_train,batch_size=10, nb_epoch = 100)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

newpred = classifier.predict(sc.transform(np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])))

y_pred = (newpred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
