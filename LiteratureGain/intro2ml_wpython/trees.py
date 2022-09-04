"""
Developer: vkyprmr
Filename: trees.py
Created on: 2020-09-04 at 16:45:27
"""
"""
Modified by: vkyprmr
Last modified on: 2020-09-04 at 16:45:28
"""


# Imports
from loaddata import LoadData
ld = LoadData()
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.tree import export_graphviz
import graphviz
import numpy as np
import pandas as pd


# Breast cancer data
X, y, fn, tn = ld.load_cancer()
X_train, X_test, y_train, y_test = train_test_split(
X, y, stratify=y, random_state=42)
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print(f"Accuracy on training set: {tree.score(X_train, y_train)}")
print(f"Accuracy on test set: {tree.score(X_test, y_test)}")
print(f'Features used: {tree.feature_importances_}')


max_depth = [1,2,3,4,5,6,7,8,9,10]
train_acc = []
test_acc = []
for m in max_depth:
    tree = DecisionTreeClassifier(max_depth=m, random_state=0)
    tree.fit(X_train, y_train)
    train = tree.score(X_train, y_train)
    test = tree.score(X_test, y_test)
    train_acc.append(train)
    test_acc.append(test)
    print(f'Max_depth: {m}\nTrain: {train}\tTest: {test}')

plt.plot(max_depth, train_acc, marker='o', label='Train')
plt.plot(max_depth, test_acc, marker='o', label='Test')
plt.legend()
plt.xlabel('Depth')
plt.ylabel('Accuracy')
for x, y, z in zip(max_depth, train_acc, test_acc):

    label = f'{y:.2f}'

    # this method is called for each point
    plt.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center', color='blue')

    label = f'{z:.2f}'

    # this method is called for each point
    plt.annotate(label, # this is the text
                 (x,z), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,-15), # distance from text to points (x,y)
                 ha='center', color='orange')                 


export_graphviz(tree, out_file="tree.dot", class_names=["malignant", "benign"],
feature_names=fn, impurity=False, filled=True)
with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)


# Feature Importance
def plot_feature_importances_cancer(model):
    n_features = X.shape[1]
    plt.barh(range(len(fn)), model.feature_importances_, align='center')
    plt.yticks(np.arange(len(fn)), fn)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")

plot_feature_importances_cancer(tree)


