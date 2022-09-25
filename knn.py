# Importing related Python libraries
import matplotlib.pyplot as plt

# Importing SKLearn clssifiers and libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from dataset import get_dataset
import sys
import os
dataset_name = 'titanic'
X, y = get_dataset(dataset_name)
os.makedirs('knn/'+dataset_name, exist_ok=True)

f = open('knn/'+dataset_name+'/out.txt', 'w')
sys.stdout = f
# print (X)
# print(y)
# Validation split
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2, random_state=2)


dataset_size = len(X_train)
testing_accuracies = []
validation_accuracies = []
train_accuracies = []
for subset in [0.2, .4, .6, .8, 1.0]:
    subset_X = X_train[:int(dataset_size*subset)]
    subset_y = y_train[:int(dataset_size*subset)]
    features_train, labels_train = subset_X, subset_y
    knn = KNeighborsClassifier()
    k_range = list(range(1,7))
    weights_options = ['uniform','distance']
    k_grid = dict(n_neighbors=k_range, weights = weights_options)
    grid = GridSearchCV(knn, k_grid, cv=5, scoring = 'accuracy', return_train_score=True)
    grid.fit(features_train, labels_train)

    print ("Best Score: ",str(grid.best_score_))
    print ("Best Parameters: ",str(grid.best_params_))
    print ("Best Estimators: ",str(grid.best_estimator_))
    validation_accuracies.append(grid.best_score_*100)
    tr_pred =  grid.predict(subset_X)
    acc_clf = metrics.accuracy_score(subset_y, tr_pred)
    print('training accuracy: ', acc_clf)
    train_accuracies.append(int(acc_clf*100))
    label_pred = grid.predict(X_test)
    acc_clf = metrics.accuracy_score(y_test,label_pred)
    print ('testing accuracy: ', str(acc_clf))
    testing_accuracies.append(int(acc_clf*100))
    
y = testing_accuracies
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
import os
plt.savefig('knn/'+dataset_name+'/knn_plot.png')


#Get Timing
import time
st = time.time()
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn.fit(features_train, labels_train)
print ('Time taken: ', time.time() - st)

f.close()