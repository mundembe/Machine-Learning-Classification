#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 22:52:15 2022

@author: masimba
"""
# <codecell>Preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Import Dataset
Dataset = pd.read_csv("Social_Network_Ads.csv")
X = Dataset.iloc[:, :-1].values
y = Dataset.iloc[:, -1].values

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                    random_state=0)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# <codecell>Model
# Train the model
classifier = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
classifier.fit(X_train, y_train)

# Predict single value
print(classifier.predict(sc.transform([[30, 87000]])))

# Predict test set results
y_pred = classifier.predict(X_test)
print(np.concatenate((y_test.reshape(len(y_test), 1),
                      y_pred.reshape(len(y_pred), 1)), axis=1))

# Evaluate the model
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))
