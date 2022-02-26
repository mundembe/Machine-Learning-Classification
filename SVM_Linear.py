#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 13:03:33 2022

@author: masimba
"""
# <codecell>Preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

Dataset = pd.read_csv("Social_Network_Ads.csv")
X = Dataset.iloc[:, :-1].values
y = Dataset.iloc[:, -1].values

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                    random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# <codecell>Model Implimentation
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train, y_train)

# Predict single value
print(classifier.predict(sc.transform([[30, 87000]])))

# Predict test set results
y_pred = classifier.predict(X_test)
print(np.concatenate((y_test.reshape(len(y_test), 1),
                      y_pred.reshape(len(y_test), 1)), 1))

# Confusion matrix and accuracy score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))
