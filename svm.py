# Importing related Python libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from dataset import get_dataset
import os
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import sys
import time 
dataset_name = 'titanic'
X, y = get_dataset(dataset_name)
src_dr = 'svm/'+dataset_name
os.makedirs('svm/'+dataset_name, exist_ok=True)
f = open('svm/'+dataset_name+'/out.txt', 'w')
sys.stdout = f
params = [['poly', 0.1], ['poly', 10], ['rbf',1], ['rbf', 10]]

for idx, param in enumerate(params):
    svm_model = make_pipeline(StandardScaler(), SVC(gamma='auto', kernel=param[0], C=param[1]))
    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(svm_model, X, y, cv=5, return_times=True)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]    
    plt.plot(train_sizes/(len(y)),np.mean(train_scores,axis=1), label='training')
    plt.plot(train_sizes/(len(y)),np.mean(test_scores,axis=1), label='testing')
    plt.legend()        
    plt.ylabel('Accuracy')
    plt.xlabel('dataset size %')
    plt.ylim(0, 1.0)

    plt.savefig(src_dr+'/paramset_'+str(idx)+'_'+dataset_name+'_nn_vs_trainsize.png')
    plt.clf()
    plt.plot(fit_time_sorted,np.mean(train_scores,axis=1), label='training')
    plt.plot(fit_time_sorted,np.mean(test_scores,axis=1), label='testing')
    plt.legend()
    plt.ylabel('Accuracy')
    plt.xlabel('Time / s')
    plt.ylim(0, 1.0)

    plt.savefig(src_dr+'/paramset_'+str(idx)+'_'+dataset_name+'_nn_vs_fittime.png')
    plt.clf()


st = time.time()
svm_model = make_pipeline(StandardScaler(), SVC(gamma='auto', kernel=param[0], C=param[1]))
svm_model.fit(X, y)
print ('Time taken to fit:', time.time() - st)
f.close()