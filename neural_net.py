# Importing related Python libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
# Importing SKLearn clssifiers and libraries
from sklearn.neural_network import MLPClassifier
from dataset import get_dataset
import os
import sys
dataset_name = 'breast_cancer'
X, y = get_dataset(dataset_name)
src_dr = 'neural_net/'+dataset_name
os.makedirs('neural_net/'+dataset_name, exist_ok=True)
f = open('neural_net/'+dataset_name+'/out.txt', 'w')
sys.stdout = f

params = [[[10],'adam'], [[10],'sgd'], [[40,40], 'adam'], [[40,40], 'sgd']]
import time
for idx, param in enumerate(params):
    mlp = MLPClassifier(solver=param[1], activation='relu', alpha=1e-5,
                            hidden_layer_sizes=param[0], random_state=2, max_iter=20000)
    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(mlp, X, y, cv=5, return_times=True)
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
    plt.ylim(0, 1.0)
    plt.legend()
    plt.ylabel('Accuracy')
    plt.xlabel('Time / s')    
    plt.savefig(src_dr+'/paramset_'+str(idx)+'_'+dataset_name+'_nn_vs_fittime.png')
    plt.clf()

st = time.time()
mlp = MLPClassifier(solver=param[1], activation='relu', alpha=1e-5,
                            hidden_layer_sizes=param[0], random_state=2, max_iter=20000)
mlp.fit(X, y)
print ('Time taken to fit:', time.time() - st)
f.close()