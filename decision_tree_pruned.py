# Importing related Python libraries
import matplotlib.pyplot as plt

# Importing SKLearn clssifiers and libraries
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from dataset import get_dataset
import os, sys

dataset_name = 'titanic'
X, y = get_dataset(dataset_name)
os.makedirs('decision_tree/'+dataset_name, exist_ok=True)
f = open('decision_tree/'+dataset_name+'/out.txt', 'w')
sys.stdout = f

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2, shuffle=True, random_state=2)

dataset_size = len(X_train)
accuracies = []
best_acc = 0
depth_range = range(1, 10)
best_depth = 1
testing_accuracies = []
validation_accuracies = []
train_accuracies = []

ccp_alphas = tree.DecisionTreeClassifier().cost_complexity_pruning_path(X_train, y_train)['ccp_alphas']
ccp_alphas=[0.003]
params = {
    "ccp_alpha":[alpha for alpha in ccp_alphas],
    'max_depth':[1,2,3,4,5,7],
    'criterion':["entropy", 'gini']}
for subset in [0.2, .4, .6, .8, 1.0]:
    subset_X = X_train[:int(dataset_size*subset)]
    subset_y = y_train[:int(dataset_size*subset)]
    features_train, labels_train = subset_X, subset_y
    
    tree_model = tree.DecisionTreeClassifier()
    
    grid = GridSearchCV(estimator=tree_model,
                    cv=5,
                    scoring=make_scorer(accuracy_score),
                    param_grid=params)
    
    grid.fit(X = features_train, 
                            y = labels_train) # We fit the model with the fold train data
    tr_pred =  grid.predict(subset_X)
    print(grid.best_params_)
    acc_clf = metrics.accuracy_score(subset_y, tr_pred)
    print('training accuracy: ', acc_clf)
    train_accuracies.append(acc_clf)
    # print(tree.plot_tree(tree_model))
    validation_accuracies.append(grid.best_score_)
    print('cross validation accuracy: ', grid.best_score_)
    test_acc = grid.score(X = X_test, 
                                    y = y_test)
    testing_accuracies.append(test_acc)
    print('testing accuracy: ', test_acc)

x = [0.2, .4, .6, .8, 1.0]
plt.plot(x, testing_accuracies, label='test')
plt.plot(x,train_accuracies, label='training')
plt.plot(x,validation_accuracies, label='validation')
# naming the x axis
plt.ylabel('Accuracy')
# naming the y axis
plt.xlabel('dataset size')
plt.title('Testing accuracy vs dataset size')
plt.legend()
plt.savefig('decision_tree/'+dataset_name+'/decision_tree_plot.png')

import time 
st = time.time()
tree_model = tree.DecisionTreeClassifier(ccp_alpha=0.013509622441443131, criterion='entropy', max_depth= 4)
tree_model.fit(X,y)
print ('Time taken to fit: ', time.time() - st)
f.close()