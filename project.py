# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 15:49:33 2019

@author: Gianluca
"""

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import time
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

ds=pd.read_csv("TrainingDataset.arff.txt",header=None, comment='@')
train, test = train_test_split(ds, test_size=0.2)
print np.shape(train),np.shape(test)

X_train = train.iloc[:,:np.shape(train)[1]-1]
Y_train = train.iloc[:,-1]
X_test = test.iloc[:,:np.shape(train)[1]-1]
Y_test = test.iloc[:,-1]
print np.shape(X_train),np.shape(Y_train),np.shape(X_test),np.shape(Y_test)

'''
C_range = np.logspace(-3 , 6 , num=10)
val_scores = []
test_scores = []
for i in C_range:
    clf = svm.SVC(C=i,kernel='linear')
    scores = cross_val_score(clf,X_train,Y_train,scoring='accuracy',cv=3)
    print "Using C=",i,"scores are:\n",scores
    print "Mean =",scores.mean()
    val_scores.append(scores.mean())
    clf.fit(X_train,Y_train)
    t_score = clf.score(X_test,Y_test)
    print t_score
    test_scores.append(t_score)
plt.plot(C_range,val_scores)
plt.plot(C_range,test_scores)
plt.xlabel("C")
plt.xscale('log')
plt.ylabel("Accuracy")
plt.legend(['Validation accuracy','Test accuracy'], prop={'size': 10})
plt.show()
'''
C_range = np.logspace(-3 , 6, num=4)
print C_range
parameters = {'C':C_range, 'kernel':['linear']}
val_scores = []
train_scores = []
test_scores = []
svc = svm.SVC()
clf = GridSearchCV(svc, parameters, cv=3, scoring="accuracy",n_jobs=-1)
clf.fit(X_train, Y_train)

for i in range(0,len(clf.cv_results_['params'])):
        print ("Modello:",clf.cv_results_['params'][i], "accuracy:",clf.cv_results_['mean_test_score'][i])


for i in range (0,len(clf.cv_results_['mean_test_score'])): 
    val_scores.append(clf.cv_results_['mean_test_score'][i])
    train_scores.append(clf.cv_results_['mean_train_score'][i])
print "val scores:",val_scores,"\ntrain scores:",train_scores
plt.plot(C_range,val_scores)
plt.plot(C_range,train_scores)
plt.xlabel("C")
plt.xscale('log')
plt.ylabel("Accuracy")
plt.legend(['Validation accuracy','Train accuracy'], prop={'size': 10})
plt.show()