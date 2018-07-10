# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 15:28:10 2016

@author: User
"""

import random, math

import scipy.io
import matplotlib as mpl
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np 
import time

import matplotlib.pyplot as plt
from sklearn import svm

from sklearn.svm import SVC
X=pd.read_csv('c:/Users/User/workspace/DAT210x/Module6/Datasets/parkinsons.data')
X=X.drop ('name',1)
y=X['status']
X=X.drop('status',1)
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=7)
svc=SVC()
svc.fit(X_train, y_train)
score=svc.score(X_test,y_test)
print score

def best_score_svc(kernel,X_train,y_train,X_test,y_test):
    best_score=0
    for c in np.arange (0.05,2.0,0.05):
            for gamma in np.arange(0.001,0.1,0.001):
                svc=SVC(kernel=kernel,C=C,gamma=gamma)
                svc.fit(X_train, y_train)
                score=svc.score(X_test,y_test)
                best_score = score if score > best_score else best_score
    return best_score
       
print best_score_svc("rbf",X_train,y_train,X_test,y_test)

    