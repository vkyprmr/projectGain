'''
Developer: vkyprmr
Filename: meet_the_data_iris.py
Created on: 2020-09-01 at 22:01:16
'''
'''
Modified by: vkyprmr
Last modified on: 2020-09-02 at 21:42:07
'''

#%%
# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib qt
import mglearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#%%
# Data
iris_dataset = load_iris()
"""
    iris_dataset is a dictionary containing the data, target, feature_names, target_names, description, filename
"""
data = iris_dataset['data']
target = iris_dataset['target']
feature_names = iris_dataset['feature_names']
target_names = iris_dataset['target_names']

# %%
# Split data
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=0, shuffle=True)

# %%
# Looking at the data
iris_df = pd.DataFrame(X_train, columns=feature_names)
## Creating a scatter matrix and color by the features
grr = pd.plotting.scatter_matrix(iris_df, c=y_train, marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)

# %%
# Frist kNN classifier
knc = KNeighborsClassifier(n_neighbors=1)

## Train (fit the model)
knc.fit(X_train, y_train)

## Predictions
preds = knc.predict(X_test)
for pred in preds:
    print(f'Prediction: {pred}, class: {target_names[pred]}')

score = np.mean(preds==y_test)
print(f'Score: {score}')
print(f'Score: {knc.score(X_test, y_test)}')

# %%
