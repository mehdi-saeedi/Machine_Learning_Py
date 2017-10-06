import pandas as pd
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV


def groupify(x):
    max_group = 5
    if x['FamilySize'] > 0:
        return x['FamilySize']
    elif x['FreqTicket'] > 1:
        return x['FreqTicket']
    elif x['FreqCabin'] > 1 and x['Cabin'] != 'U':
        return x['FreqCabin']
    elif 1 < x['FreqLastName'] < max_group:
        return x['FreqLastName']
    elif 1 < x['FreqFare'] < max_group:
        return x['FreqFare']
    else:
        return 0


directory = '../../Datasets/Titanic/'

# get the dataset
titanic_train = pd.read_csv(directory + 'train.csv')
titanic_test = pd.read_csv(directory + 'test.csv')
titanic_train_test = [titanic_train, titanic_test]

Title_Dictionary = {
    "Capt": "Rare",
    "Col": "Rare",
    "Major": "Rare",
    "Jonkheer": "Rare",
    "Don": "Rare",
    "Sir": "Rare",
    "Dr": "Rare",
    "Rev": "Rare",
    "the Countess": "Rare",
    "Dona": "Rare",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr": "Mr",
    "Mrs": "Mrs",
    "Miss": "Miss",
    "Master": "Master",
    "Lady": "Rare"
}

for dataset in titanic_train_test:
    dataset['Age'].fillna(dataset.Age.median(), inplace=True)
    dataset['Cabin'].fillna('U', inplace=True)
    dataset['Embarked'].fillna('S', inplace=True)
    dataset['Fare'].fillna(dataset.Fare.mean(), inplace=True)

    dataset['FamilySize'] = dataset.SibSp + dataset.Parch
    dataset['AgeRange'], AgeBins = pd.cut(dataset['Age'], 10, retbins=True)

    dataset['Title'] = dataset['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())
    dataset['Title'] = dataset['Title'].map(Title_Dictionary)

    dataset['LastName'] = dataset['Name'].map(lambda name: name.split(',')[0].strip())

    for col in ['Ticket', 'Cabin', 'Fare', 'LastName']:
        freq_col = f'Freq{col}'

        freq = dataset[col].value_counts().to_frame()
        freq.columns = [freq_col]

        dataset[freq_col] = dataset.merge(freq, how='left', left_on=col, right_index=True)[freq_col]

    # Maximum size of groups that share a fare value.
    dataset['GroupSize'] = dataset.apply(groupify, axis=1)

print(titanic_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())
print()
print(titanic_train[['GroupSize', 'Survived']].groupby(['GroupSize'], as_index=False).mean())
print()
print(titanic_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())
print()
print(titanic_train[['AgeRange', 'Survived']].groupby(['AgeRange'], as_index=False).mean())
print()
print(titanic_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())
print()

y = titanic_train['Survived']
titanic_train.drop(['Survived'], axis=1, inplace=True)

for dataset in titanic_train_test:
    dataset['Title'] = dataset['Title'].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}).astype(int)

    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    for AgeGroup in range(0, len(AgeBins)):
        if AgeGroup == len(AgeBins) - 1:
            dataset.loc[dataset['Age'] > AgeBins[AgeGroup], 'Age'] = AgeGroup
        else:
            dataset.loc[
                (dataset['Age'] > AgeBins[AgeGroup]) & (dataset['Age'] <= AgeBins[AgeGroup + 1]), 'Age'] = AgeGroup

    dataset["Pclass"] = dataset["Pclass"].astype('int')

    # Sex & Title have correclation. We keep Title.
    for col in dataset.columns:
        if col not in ['Pclass', 'Age', 'Embarked', 'Title', 'GroupSize']:
            dataset.drop([col], inplace=True, axis=1)
    for col in dataset.columns:
        dataset[col] = dataset[col].astype("category")

titanic_train = pd.get_dummies(titanic_train, columns=None)
titanic_test = pd.get_dummies(titanic_test, columns=None)
missing_cols = set(titanic_train.columns) - set(titanic_test.columns)
for c in missing_cols:
    titanic_test[c] = 0
missing_cols = set(titanic_test.columns) - set(titanic_train.columns)
for c in missing_cols:
    titanic_test[c] = 0

X_train, y_train = titanic_train, y
X_test = titanic_test

# Set the parameters by cross-validation
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

# run svm
C_range = np.logspace(-3, 3, 7)
gamma_range = np.logspace(-3, 3, 7)
param_grid = dict(gamma=gamma_range, C=C_range)
svm_model = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
svm_model.fit(X_train, y_train)

print("[SVM] The best parameters are %s with a score of %0.2f"
      % (svm_model.best_params_, svm_model.best_score_))

# MLP
alpha_range = np.logspace(-3, 3, 7)
param_grid = dict(alpha=alpha_range)
mlp = GridSearchCV(MLPClassifier(solver='lbfgs'), param_grid=param_grid, cv=cv)
mlp.fit(X_train, y_train)

print("[MLP] The best parameters are %s with a score of %0.2f"
      % (mlp.best_params_, mlp.best_score_))

# Tree
max_depth_range = np.linspace(10, 15, 6).astype(int)
min_samples_split_range = np.linspace(2, 5, 4).astype(int)
param_grid = dict(max_depth=max_depth_range, min_samples_split=min_samples_split_range)
clf = GridSearchCV(DecisionTreeClassifier(), param_grid=param_grid, cv=cv)
clf.fit(X_train, y_train)

print("[TREE] The best parameters are %s with a score of %0.2f"
      % (clf.best_params_, clf.best_score_))

# Random Forest
param_grid = {"n_estimators": [250, 300],
              "criterion": ["gini", "entropy"],
              "max_depth": [10, 15, 20],
              "min_samples_split": [2, 3, 4]}
forest = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=cv, verbose=1)
forest.fit(X_train, y_train)

print("[FOREST] The best parameters are %s with a score of %0.2f"
      % (forest.best_params_, forest.best_score_))

out_df = pd.DataFrame(
    {'PassengerId': pd.read_csv(directory + 'test.csv').PassengerId, 'Survived': svm_model.predict(X_test)}).to_csv(
    'out_svm.csv', header=True, index=False)

out_df = pd.DataFrame(
    {'PassengerId': pd.read_csv(directory + 'test.csv').PassengerId, 'Survived': mlp.predict(X_test)}).to_csv(
    'out_mlp.csv', header=True, index=False)

out_df = pd.DataFrame(
    {'PassengerId': pd.read_csv(directory + 'test.csv').PassengerId, 'Survived': clf.predict(X_test)}).to_csv(
    'out_tree.csv', header=True, index=False)

out_df = pd.DataFrame(
    {'PassengerId': pd.read_csv(directory + 'test.csv').PassengerId, 'Survived': forest.predict(X_test)}).to_csv(
    'out_forest.csv', header=True, index=False)


plt.show()
