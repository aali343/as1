import pandas as pd

def get_dataset(name):
    if name=='titanic':
        df_train = pd.read_csv('data/train.csv')
        df_train.sample(frac=1)
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
        x_train.head()
        x_train['Age'] = x_train['Age'].fillna(x_train['Age'].median())
        x_train = x_train[x_train['Embarked'].notna()]
        X = x_train.copy()
        X = X.drop(columns=['Survived'])
        df2 = x_train['Survived']

        y = df2
        return X, y
    elif name=='breast_cancer':
        df = pd.read_csv('data/breastcancerdata.csv')
        df.sample(frac=1)
        X=df[["radius_mean","texture_mean","smoothness_mean","compactness_mean","concavity_mean"]]
        X=(X-X.min())/(X.max()-X.min())
        df.loc[df["diagnosis"] == "M", "diagnosis"] = 0
        df.loc[df["diagnosis"] == "B", "diagnosis"] = 1
        y=df["diagnosis"]
        return X.to_numpy(),y.to_numpy().tolist()

get_dataset('breast_cancer')