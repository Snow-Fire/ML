# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 23:35:22 2019

@author: Ullas
"""
#import MNIST data
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
mnist

#checking the size of the data 
X, y = mnist["data"], mnist["target"]
X.shape
y.shape

import matplotlib
import matplotlib.pyplot as plt
#checking for any random digit
some_digit = X[36000]
some_digit_image = some_digit.reshape(28,28)
plt.imshow(some_digit_image, cmap = matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()
y[36000]

#Separate out the data into training and test data
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

#shuffling
import numpy as np
shuffle_index = np.random.permutation(1000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

#Boolean arrays that have True for all values that are 5 else False
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

#Stochastic Gradient Descent Classifier
#Not running for the momemnt some error. Need to check on this later START
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

sgd_clf.predict([some_digit])

from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

#Confustion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_5, y_train_pred)

#Precision and Recall Score
from sklearn.metrics import precision_score, recall_score
precision_score(y_train_5, y_train_pred)
recall_score(y_train_5, y_train_pred)

#F1 Score
from sklearn.metrics import f1_score
f1_score(y_train_5, y_train_pred)

#Decision Score
#For Instance
y_scores = sgd_clf.decision_function([some_digit])
y_scores
#For whole dataset
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,method="decision_function")
y_scores
y_scores.shape

#Precision Recall Curve
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
plt.xlabel("Threshold")
plt.legend(loc="upper left")
plt.ylim([0, 1])
plt.show()

#Precision Against Recall
plt.plot(recalls, precisions, "b-", linewidth=2)
plt.xlabel("Recall", fontsize=16)
plt.ylabel("Precision", fontsize=16)
plt.axis([0, 1, 0, 1])
plt.show()

#ROC
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

plt.plot(fpr, tpr, linewidth=2)
plt.plot([0, 1], [0, 1], 'k--')
plt.axis([0, 1, 0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

#Area Under Curve
from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5, y_scores)

#Not running for the momemnt some error. Need to check on this later END
