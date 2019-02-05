import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import time
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


start_time= time.time()

#SEZIONE 1: Lettura dati
ds=pd.read_csv("TrainingDataset.arff.txt",header=None, comment='@')

#Shuffle per stratified sampling e splitting dei dati
ds_shuff=shuffle(ds)
positive= ds[ds.iloc[:,-1]>0]
negative= ds[ds.iloc[:,-1]<0]
X = ds_shuff.iloc[:, :ds.shape[1]-1]
Y = ds_shuff.iloc[:,-1]
n=len(ds)
nl=int(round(.9*n))

X_train = ds_shuff.iloc[:nl,:ds.shape[1]-1]
Y_train = ds_shuff.iloc[:nl,-1]
X_test = ds_shuff.iloc[nl:,:ds.shape[1]-1]
Y_test = ds_shuff.iloc[nl:,-1]

X_selected=[]

#Plotting delle features
'''
for i in range(0,ds.shape[1]-1):
    xp = positive.iloc[:,i]
    xn = negative.iloc[:,i]
    colors = ['blue', 'red']
    plt.hist([xp,xn], bins=[-1.5,-0.5,0.5,1.5],stacked=True, color=colors, label=['positive','negative'])
    plt.legend(prop={'size': 10})
    plt.xlabel('Values')
    plt.ylabel('Samples')
    plt.title('Histogram of feature %s'%i)
    plt.axis([-1.5, 1.5, 0, len(ds)])
    plt.grid(True)
    plt.show()
'''

corr= ds_shuff.corr().abs()
columns= np.full((corr.shape[0],),True, dtype=bool)
for i in range(corr.shape[0]):
    '''
    #elimino le feature meno correlate con l'output (ne rimangono 12 con questa soglia)
    if corr.iloc[i,corr.shape[0]-1]<0.1:
        columns[i]=False
        
    #Miglior tune: {'C': 100.0, 'gamma': 0.1, 'kernel': 'rbf'} 
    #con media: 0.9506105834464044 +/-( 0.012578220480502866 )
    #Accuratezza del miglior modello su tutto il dataset: 0.965445499773858
    '''   
    for j in range(i+1, corr.shape[0]-1):
        if corr.iloc[i,j]>=0.9:
            if columns[j] and columns[i]:
                if corr.iloc[i,corr.shape[0]-1]>corr.iloc[j,corr.shape[0]-1]:
                    columns[j]=False
                else:
                    columns[i]=False
                
#selected_columns=ds_shuff.columns[columns]
#X_selected= ds_shuff[selected_columns]
#X_selected=X_selected.iloc[:,:X_selected.shape[1]-1]
                    
#elimina la 22esima feature(indice 21)
#utilizzando questo dataset il miglior modello rimane lo stesso ma l'accuratezza scende leggermente
#(comunque meglio rispetto a feature selection in base alla varianza)        
#Miglior tune: {'C': 10.0, 'gamma': 0.1, 'kernel': 'rbf'} 
#con media: 0.970239710538218 +/-( 0.009364237624589815 )
#Accuratezza del miglior modello su tutto il dataset: 0.9843509724106739

for i in range(0,ds.shape[1]-1):
    variance = np.var(ds.iloc[:,i])
    #selezione feature in base alla varianza
    if(variance<0.2):
        columns[i]=False
        
selected_columns=ds_shuff.columns[columns]
X_selected= ds_shuff[selected_columns]     
X_selected=X_selected.iloc[:,:X_selected.shape[1]-1]

#elimina la 19esima e la 21esima feature(indice 18 e 20)      
#utilizzando questo dataset il miglior modello rimane lo stesso ma l'accuratezza scende leggermente        
#Miglior tune: {'C': 10.0, 'gamma': 0.1, 'kernel': 'rbf'} 
#con media: 0.9707824513794663 +/-( 0.009174944347157835 )
#Accuratezza del miglior modello su tutto il dataset: 0.9841700587969244


#eliminando entrambi i tipi di features abbiamo ottenuto un risultato meno accurato
#Miglior tune: {'C': 10.0, 'gamma': 0.1, 'kernel': 'rbf'} 
#con media: 0.9703301673450927 +/-( 0.009446480516524846 )
#Accuratezza del miglior modello su tutto il dataset: 0.9840796019900497

#SEZIONE 2: Single accuracy con parametro fisso

clf = svm.SVC(kernel='linear',C=1)
clf.fit(X_train,Y_train)
single_accuracy = clf.score(X_test,Y_test)
print ("\nAccuratezza ottenuta con %s campioni per il training su %s: %s e C=1" % (nl,n,single_accuracy))
#Accuratezza ottenuta con 9950 campioni per il training su 11055: tra 0.92 e 0.94 circa

#SEZIONE 3: K-fold Cross-Validation con parametro fisso
'''
print ("\nCalcolo accuratezza con 10-fold cross-validation e C=1...")
kclf = svm.SVC(kernel='linear',C=1)
kfoldscores = cross_val_score(kclf,X,Y,cv=10,scoring = 'accuracy')
print (kfoldscores)
kmean=kfoldscores.mean()
kstd=kfoldscores.std()
print ("In media: %s +/-(%s)" % (kmean,kstd))
#In media: 0.9273622048201187 +/-(0.006292528176258127)
'''
#SEZIONE 4: K-fold Cross-Validation con parametro variabile
'''
for i in range(0,3):
    print("\nCalcolo accuratezza con 10-fold cross-validation e C variabile")
    C_best=None
    kmean_best=0
    kstd_best=np.inf
    k_scores = [] #lista per i K valori medi
    k_stds = [] #lista per le K deviazioni standard
    C_range = np.logspace(-1 , 2 , num=4)
    
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
    plt.xscale('log')
    plt.ylabel("Cross-Validated Accuracy")
 
    print ("Miglior risultato ottenuto con C=%s e accuratezza media=%s +/-(%s)"%(C_best,kmean_best,kstd_best))
    #Miglior risultato con C=10 e accuratezza media 0.927362 +/-(0.006240)
    ds_shuff=shuffle(ds)
    X = ds_shuff.iloc[:, :ds.shape[1]-1]
    Y = ds_shuff.iloc[:,-1]
plt.title("Kernel lineare")
plt.show()  
 
#SEZIONE 5: GridSearchCV tuning
'''
for i in range(0,6):
    print("\nScelta del kernel e tuning dei parametri C e Gamma con GridSearchCV...")
    C_range = np.logspace(-2 , 2 , num=5)
    Gamma_range= np.logspace(-5, 1, num=7) 
    param_grid = [
     # {'C': C_range, 'kernel': ['linear']},
      {'C': C_range, 'gamma': Gamma_range, 'kernel': ['rbf']},
     ]
    
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
    
    #Plot dei modelli
    i=0
    while(i<len(clf.cv_results_['params'])):
        
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


#SEZIONE 6: training finale
model = clf.best_estimator_
model.fit(X,Y)
print("\nAccuratezza del miglior modello su tutto il dataset:",model.score(X,Y))
#Miglior tune: {'C': 10.0, 'gamma': 0.1, 'kernel': 'rbf'} 
#con media: 0.970872908186341 +/-( 0.009391939891934063 )

#Modello completo: 
# SVC(C=10.0, cache_size=200, class_weight=None, coef0=0.0,
# decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',
#  max_iter=-1, probability=False, random_state=None, shrinking=True,
#  tol=0.001, verbose=False)

#Accuratezza del miglior modello su tutto il dataset: 0.9844414292175486

elapsed_time=time.time()-start_time
print("\nElapsed time:",elapsed_time)
#Elapsed time: 679.6059725284576