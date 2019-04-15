# gender classifier - https://github.com/Naresh1318/GenderClassifier/blob/master/Run_Code.py

from sklearn import tree 
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np


# Data set = [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]
     
Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

# listing the classfiers
clf_tree = tree.DecisionTreeClassifier()
clf_svm =  SVC(gamma="auto")
clf_perceptron = Perceptron()
clf_KNN = KNeighborsClassifier()
clf_NB = GaussianNB()

# training the models
clf_tree = clf_tree.fit(X,Y)
clf_svm = clf_svm.fit(X,Y)
clf_perceptron = clf_perceptron.fit(X,Y)
clf_KNN = clf_KNN.fit(X,Y)
clf_NB = clf_NB.fit(X,Y)

# Testing using the above data 
pred_tree = clf_tree.predict(X)
accuracy_tree = accuracy_score(Y, pred_tree)*100
print ('Accuracy for DecisionTree: {}'.format(accuracy_tree))

pred_svm = clf_svm.predict(X)
accuracy_svm = accuracy_score(Y, pred_svm)*100
print ('Accuracy for SVM: {}'.format(accuracy_svm))

pred_per = clf_perceptron.predict(X)
accuracy_per = accuracy_score(Y, pred_per)*100
print ('Accuracy for Perceptron: {}'.format(accuracy_per))

pred_KNN = clf_KNN.predict(X)
accuracy_KNN = accuracy_score(Y, pred_KNN)*100
print ('Accuracy for KNN: {}'.format(accuracy_KNN))

pred_NB = clf_NB.predict(X)
accuracy_NB = accuracy_score(Y, pred_KNN)*100
print ('Accuracy for NB: {}'.format(accuracy_NB))

# Best classfier among all
index = np.argmax([accuracy_svm, accuracy_per, accuracy_KNN, accuracy_NB])
classifiers = {0: 'SVM', 1: 'Perceptron', 2:'KNN', 3:'NB'}
print ('Best gender classifier is {}'.format(classifiers[index]))








