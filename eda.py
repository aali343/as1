# Importing related Python libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv

# Importing SKLearn clssifiers and libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
df_train = pd.read_csv('data/train.csv')

print(df_train)
#Drop features that seem irrelevant to survival rte
df_train = df_train.drop('Name', axis=1,)
df_train = df_train.drop('Ticket', axis=1,)
df_train = df_train.drop('Fare', axis=1,)
df_train = df_train.drop('Cabin', axis=1,)
df_train = df_train.drop('PassengerId', axis=1,)
x_train = df_train.copy()
x_train.loc[df_train["Embarked"] == "S", "Embarked"] = 0
x_train.loc[df_train["Embarked"] == "C", "Embarked"] = 1
x_train.loc[df_train["Embarked"] == "Q", "Embarked"] = 2
x_train.loc[df_train["Sex"] == "male", "Sex"] = 0
x_train.loc[df_train["Sex"] == "female", "Sex"] = 1


print('Number of empty Age entrees', x_train['Age'].isna().sum())
print('Number of empty Embarked entrees', x_train['Embarked'].isna().sum())