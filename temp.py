# -- coding: utf-8 --
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import time
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
'''
def doPCA(X):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(X)
    return pca
'''
start_time= time.time()

#SEZIONE 1: Lettura dati e preprocessing

#ds.read_csv("data_banknote_authentication.txt",header=None)
ds=pd.read_csv("TrainingDataset.arff.txt",header=None)

#Shuffle per stratified sampling
ds_shuff=shuffle(ds)
positive= ds[ds.iloc[:,-1]>0]
negative= ds[ds.iloc[:,-1]<0]
X = ds.iloc[:, :ds.shape[1]-1]
Y = ds.iloc[:,-1]


# the histogram of the data
for i in range(0,ds.shape[1]-1):
    xp = positive.iloc[:,i]
    xn = negative.iloc[:,i]
    colors = ['blue', 'red']
    n, bins, patches = plt.hist([xp,xn], bins=[-1.5,-0.5,0.5,1.5],stacked=True, color=colors, label=['positive','negative'])
    plt.legend(prop={'size': 10})
    plt.xlabel('Values')
    plt.ylabel('Samples')
    plt.title('Histogram of feature %s'%i)
    plt.axis([-1.5, 1.5, 0, len(ds)])
    plt.grid(True)
   

    plt.show()


#SEZIONE 2: Splitting dei dati

n=len(ds)
nl=int(round(.9*n))

X_train = ds_shuff.iloc[:nl,:ds.shape[1]-1]
Y_train = ds_shuff.iloc[:nl,-1]
X_test = ds_shuff.iloc[nl:,:ds.shape[1]-1]
Y_test = ds_shuff.iloc[nl:,-1]

#SEZIONE 3: Ulteriori trasformazioni
'''
pca = doPCA(X)
print pca.explained_variance_ratio_
first_pc = pca.components_[0]
second_pc = pca.components_[1]

transformed_X = pca.transform(X)
for ii, jj in zip(transformed_X,X):
    plt.scatter(first_pc[0]*ii[0],first_pc[1]*ii[0],color='r')
    plt.scatter(second_pc[0]*ii[1],second_pc[1]*ii[1],color='c')
    plt.scatter(jj[0],jj[1],color='b')
    
plt.show()
'''

#SEZIONE 4: Single accuracy con parametro fisso
clf = svm.SVC(kernel='linear',C=1)
clf.fit(X_train,Y_train)
single_accuracy = clf.score(X_test,Y_test)
print ("Accuratezza ottenuta con %i campioni per il training su %i: %s" % (nl,n,single_accuracy))

#SEZIONE 4: K-fold Cross-Validation con parametro fisso

print ("Calcolo accuratezza con 10-fold cross-validation...")
kclf = svm.SVC(kernel='linear',C=1)
kfoldscores = cross_val_score(kclf,X,Y,cv=10,scoring = 'accuracy')
print (kfoldscores)
kmean=kfoldscores.mean()
kstd=kfoldscores.std()
print ("In media: %s +/-(%s)" % (kmean,kstd))

#SEZIONE 5: K-fold Cross-Validation con parametro variabile

C_best=None
kmean_best=0
kstd_best=np.inf
k_scores = [] #lista per i K valori medi
k_stds = [] #lista per le K deviazioni standard
C_range = np.logspace(-1 , 1 , num=3)
'''
for C in C_range:
    print ("\nCalcolo accuratezza per C=%s" % C)
    kclf = svm.SVC(kernel='linear',C=C)
    kfoldscores = cross_val_score(kclf,X,Y,cv=10,scoring = 'accuracy')
    print (kfoldscores)
    kmean=kfoldscores.mean()
    k_scores.append(kmean)
    kstd=kfoldscores.std()
    k_stds.append(kstd)
    print ("In media: %s +/-(%s)" % (kmean,kstd)  ) 
    if (kmean>kmean_best):
        kmean_best = kmean
        kstd_best = kstd
        C_best = C
    elif (kmean==kmean_best):
        if (kstd<kstd_best):
            kstd_best=kstd
            C_best = C
        
plt.plot(C_range,k_scores)
plt.xlabel("Valore di C")
plt.ylabel("Cross-Validated Accuracy")
'''
#SEZIONE 6: GridSearchCV tuning
C_range = np.logspace(-2 , 2 , num=5)
Gamma_range= np.logspace(-5, 1, num=7)
 
param_grid = [
  {'C': C_range, 'kernel': ['linear']},
  {'C': C_range, 'gamma': Gamma_range, 'kernel': ['rbf']},
 ]

#parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = svm.SVC()
clf = GridSearchCV(svc, param_grid, cv=10, scoring="accuracy", n_jobs=-1)
clf.fit(X, Y)
print ("")

for i in range(0,len(clf.cv_results_['params'])):
    print ("Modello:",clf.cv_results_['params'][i], "accuracy:",clf.cv_results_['mean_test_score'][i])

accuracy=[]
label=[]
for i in C_range:
    label.append("C="+str(i))
i=0
while(i<len(clf.cv_results_['params'])):
    if clf.cv_results_['params'][i]['kernel']=='linear':
        for j in range(i, i+len(C_range)):
            accuracy.append(clf.cv_results_['mean_test_score'][j])
        i= i+len(C_range)
        plt.plot(C_range,accuracy)
        plt.xscale('log')
        plt.xlabel("Valore di C")
        plt.ylabel("Accuratezza")
        plt.title("Kernel lineare")
        plt.show()
    elif clf.cv_results_['params'][i]['kernel']=='rbf':
        maxdim= max(len(C_range), len(Gamma_range))
        c=clf.cv_results_['params'][i]['C']
        for j in range(i, i+maxdim):
            accuracy.append(clf.cv_results_['mean_test_score'][j])
        i=i+maxdim
        plt.plot(Gamma_range,accuracy)
        plt.xscale('log')
        plt.xlabel("Valore di gamma")
        plt.ylabel("Accuratezza")
        
    accuracy=[]    
plt.legend(label, prop={'size': 10})
plt.title("Kernel RBF")
plt.show()     




print ("\nMiglior tune:" , clf.best_params_ , "\ncon media:" , clf.best_score_ , "+/-(" , clf.cv_results_['std_test_score'][clf.best_index_] , ")")
print ("\nModello completo: \n",  clf.best_estimator_)

#support vectors divisi per support hyperplane
supp_vect = clf.best_estimator_.n_support_
#insieme dei support vectors
supportvectors = clf.best_estimator_.support_vectors_
#alpha_i * y_i con i t.c alpha_i != 0 (alpha compreso tra [0,C])
dual_coeff = clf.best_estimator_.dual_coef_
if clf.best_estimator_.kernel=='linear':
    #coefficienti del vettore ortogonale all'hyperplane che separa le classi
    w_coef = clf.best_estimator_.coef_
#bias dell'hyperplane
bias = clf.best_estimator_.intercept_


#SEZIONE 7: training finale
model = clf.best_estimator_
model.fit(X,Y)
print("\nAccuratezza del miglior modello su tutto il dataset:",model.score(X,Y))

elapsed_time=time.time()-start_time
print("\nElapsed time:",elapsed_time)