# Importing related Python libraries
import numpy as np
import matplotlib.pyplot as plt

# Importing SKLearn clssifiers and libraries
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from dataset import get_dataset
import os, sys
dataset_name = 'titanic'
X, y = get_dataset(dataset_name)
os.makedirs('boosting/'+dataset_name, exist_ok=True)
f = open('boosting/'+dataset_name+'/out.txt', 'w')
sys.stdout = f

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2, shuffle=True, random_state=2)

accuracies = []
best_acc = 0
depth_range = range(1, 10)
best_depth = 1

parameters = {
    "loss":["exponential"],
    "learning_rate": [0.2], #[0.01, 0.1, 0.2, 0.5],
    "min_samples_split": [0.1],#np.linspace(0.1, 0.5, 5),
    "min_samples_leaf": [0.1], #np.linspace(0.1, 0.5, 5),
    "max_depth":[6],#, 2, 3, 4, 6, 9, 20],
    "subsample": [0.9],#, 0.5,0.2,0.1],
    "n_estimators":[50]#,20,50],
    }
dataset_size = len(X_train)
accuracies = []
best_acc = 0
depth_range = range(1, 10)
best_depth = 1
testing_accuracies = []
validation_accuracies = []
train_accuracies = []
#passing the scoring function in the GridSearchCV
for subset in [0.2, .4, .6, .8, 1.0]:
    subset_X = X_train[:int(dataset_size*subset)]
    subset_y = y_train[:int(dataset_size*subset)]
    features_train, labels_train = subset_X, subset_y
        
    grid = GridSearchCV(GradientBoostingClassifier(), param_grid = parameters,cv=5, n_jobs=-1)

    grid.fit(subset_X, subset_y)
    print(grid.best_params_)
    # for e in grid.cv_results_:
    #     print(e)
    tr_pred =  grid.predict(subset_X)
    acc_clf = metrics.accuracy_score(subset_y, tr_pred)
    print('training accuracy: ', acc_clf)
    train_accuracies.append(int(acc_clf*100))
    # print(tree.plot_tree(tree_model))
    validation_accuracies.append(grid.best_score_*100)
    print('cross validation accuracy: ', grid.best_score_*100)
    test_acc = grid.score(X = X_test, 
                                    y = y_test)
    testing_accuracies.append(test_acc*100)
    print('testing accuracy: ', test_acc*100)
    

x = [0.2, .4, .6, .8, 1.0]
plt.plot(x, testing_accuracies, label='test')
plt.plot(x,train_accuracies, label='training')
plt.plot(x,validation_accuracies, label='validation')
# naming the x axis
plt.ylabel('Accuracy')
# naming the y axis
plt.xlabel('dataset size %')
plt.title('Testing accuracy vs dataset size')
plt.legend()
import os
plt.savefig('boosting/'+dataset_name+'/boosting_plot.png')

import time 
st = time.time()
tree_model = GradientBoostingClassifier(learning_rate= 0.5, loss='exponential', max_depth=9, min_samples_leaf=0.1, min_samples_split=0.4, n_estimators=20, subsample=0.9)
tree_model.fit(X,y)
print ('Time taken to fit: ', time.time() - st)
f.close()