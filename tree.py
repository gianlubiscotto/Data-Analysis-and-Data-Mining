import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import time
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.metrics import accuracy_score
import graphviz 


start_time= time.time()

#SEZIONE 1: Lettura dati
ds=pd.read_csv("TrainingDataset.arff.txt",header=None, comment='@')
X = ds.iloc[:, :ds.shape[1]-1]
Y = ds.iloc[:,-1]
n=len(ds)
nl=int(round(.9*n))
ds_shuff=shuffle(ds)
X_train = ds_shuff.iloc[:nl,:ds.shape[1]-1]
Y_train = ds_shuff.iloc[:nl,-1]
X_test = ds_shuff.iloc[nl:,:ds.shape[1]-1]
Y_test = ds_shuff.iloc[nl:,-1]

clf = tree.DecisionTreeClassifier(max_depth=30)
clf = clf.fit(X_train, Y_train)
predictions= clf.predict(X_test)
print("Accuracy:",accuracy_score(Y_test,predictions))
print (clf.tree_.max_depth)


dot_data = tree.export_graphviz(clf, out_file=None,class_names=True,filled=True) 
graph = graphviz.Source(dot_data) 
graph.render("tree") 


#SEZIONE 3: K-fold Cross-Validation con parametro fisso

print ("\nCalcolo accuratezza con 10-fold cross-validation")
clf = tree.DecisionTreeClassifier()
kfoldscores = cross_val_score(clf,X,Y,cv=10,scoring = 'accuracy')
print (kfoldscores)
kmean=kfoldscores.mean()
kstd=kfoldscores.std()
print ("In media: %s +/-(%s)" % (kmean,kstd))

depth_range= range(1,100) 
parameters = {'max_depth': depth_range}

t = tree.DecisionTreeClassifier()
clf = GridSearchCV(t, parameters, cv=10, scoring="accuracy", n_jobs=-1)
clf.fit(X, Y)
print ("")
print(clf.cv_results_['param_max_depth'])
for i in range(0,len(clf.cv_results_['params'])):
    print ("Modello:",clf.cv_results_['params'][i], "accuracy:",clf.cv_results_['mean_test_score'][i])
