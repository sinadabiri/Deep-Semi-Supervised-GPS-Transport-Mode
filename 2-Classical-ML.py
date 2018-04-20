import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import random
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier

np.random.seed(7)
random.seed(7)


filename_train = '../Mode-codes-Revised/2_Hand_Crafted_features_filtered_train.csv'
filename_val = '../Mode-codes-Revised/2_Hand_Crafted_features_filtered_val.csv'
filename_test = '../Mode-codes-Revised/2_Hand_Crafted_features_filtered_test.csv'

prop = 1
df_train = pd.read_csv(filename_train)
X_train = np.array(df_train.loc[:, df_train.columns != 'Label'])
y_train = np.array(df_train['Label'])
index = np.arange(len(X_train))
X_train = X_train[index[:round(prop*len(X_train))]]
y_train = y_train[index[:round(prop*len(y_train))]]

df_val = pd.read_csv(filename_val)
X_val = np.array(df_val.loc[:, df_val.columns != 'Label'])
y_val = np.array(df_val['Label'])

df_test = pd.read_csv(filename_test)
X_test = np.array(df_test.loc[:, df_test.columns != 'Label'])
y_test = np.array(df_test['Label'])


# Decision Tree Grid Search
DT = DecisionTreeClassifier()
# parameters = {'max_depth': [1, 5, 10, 15, 20, 25, 30, 35, 40]}
parameters = {'max_depth': [None]}
clf = GridSearchCV(estimator=DT, param_grid=parameters, cv=5)
fit = clf.fit(X_train, y_train)
print('optimal parameter value: ', fit.best_params_)
Prediction_DT = fit.best_estimator_.predict(X_test)
Accuracy_DecisionTree = len(np.where(Prediction_DT == y_test)[0]) * 1. / len(y_test)
print('Accuracy Decision Tree: ', Accuracy_DecisionTree)
print(classification_report(y_test, Prediction_DT, digits=3))

# SVM Grid Search
SVM = SVC()
parameters = {'C': [1]}
clf = GridSearchCV(estimator=SVM, param_grid=parameters, cv=5)
fit = clf.fit(X_train, y_train)
print('optimal parameter value: ', fit.best_params_)
Prediction_SVM = fit.best_estimator_.predict(X_test)
Accuracy_SVM = len(np.where(Prediction_SVM == y_test)[0]) * 1. / len(y_test)
print('Accuracy SVM: ', Accuracy_SVM)
print(classification_report(y_test, Prediction_SVM, digits=3))

# Multilayer perceptron
MLP = MLPClassifier(early_stopping=True, hidden_layer_sizes=(2 * np.shape(X_train)[1],))
parameters = {'hidden_layer_sizes': [(2 * np.shape(X_train)[1],)]}
clf = GridSearchCV(estimator=MLP, param_grid=parameters, cv=5)
fit = clf.fit(X_train, y_train)
print('optimal parameter value: ', fit.best_params_)
Prediction_MLP = fit.best_estimator_.predict(X_test)
Accuracy_MLP = len(np.where(Prediction_MLP == y_test)[0]) * 1. / len(y_test)
print('Accuracy MLP: ', Accuracy_MLP)
print(classification_report(y_test, Prediction_MLP, digits=3))

# KNN Grid Search
KNN = KNeighborsClassifier()
parameters = {'n_neighbors': [5]}
clf = GridSearchCV(estimator=KNN, param_grid=parameters, cv=5)
fit = clf.fit(X_train, y_train)
print('optimal parameter value: ', fit.best_params_)
Prediction_KNN = fit.best_estimator_.predict(X_test)
Accuracy_KNN = len(np.where(Prediction_KNN == y_test)[0]) * 1. / len(y_test)
print('Accuracy KNN: ', Accuracy_KNN)
print(classification_report(y_test, Prediction_KNN, digits=3))
