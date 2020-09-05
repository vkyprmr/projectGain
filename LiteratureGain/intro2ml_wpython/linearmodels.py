'''
Developer: vkyprmr
Filename: linearmodels.py
Created on: 2020-09-03 at 18:52:22
'''
'''
Modified by: vkyprmr
Last modified on: 2020-09-04 at 16:45:56
'''

#%%
# Imports
from loaddata import LoadData
import matplotlib.pyplot as plt
%matplotlib qt
import mglearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.datasets import make_blobs
import numpy as np
import pandas as pd

#%%
# Loading data
ld = LoadData()

#%%
# Linear model for wave data
X, y = ld.load_wave(n_samples=60)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
lr = LinearRegression().fit(X_train, y_train)
print(f'Train score: {lr.score(X_train,y_train)}')
print(f'Test score: {lr.score(X_test,y_test)}')

# %%
# Linear model on Boston Housing data
X, y = ld.load_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
lr = LinearRegression().fit(X_train, y_train)
print(f'Train score: {lr.score(X_train,y_train)}')
print(f'Test score: {lr.score(X_test,y_test)}')

# %%
# Ridge (uses L2 regularization) model on Boston Housing data
X, y = ld.load_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
lr = Ridge().fit(X_train, y_train)
print(f'Train score: {lr.score(X_train,y_train)}')
print(f'Test score: {lr.score(X_test,y_test)}')

# Finding optimal alpha for ridge regression
train_acc = []
test_acc = []
alpha = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
for a in alpha:
    lr = Ridge(alpha=a).fit(X_train, y_train)
    train = lr.score(X_train,y_train)
    test = lr.score(X_test,y_test)
    print(f'Alpha: {a}\nTrain: {train}\tTest: {test}')
    train_acc.append(train)
    test_acc.append(test)

plt.plot(alpha, train_acc, label='Train')
plt.plot(alpha, test_acc, label='Test')
plt.legend()

# %%
# Lasso (uses L1 regularization) model on Boston Housing data
X, y = ld.load_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
lr = Lasso().fit(X_train, y_train)
print(f'Train score: {lr.score(X_train,y_train)}')
print(f'Test score: {lr.score(X_test,y_test)}')

# Finding optimal alpha for lasso regression
train_acc = []
test_acc = []
alpha = [0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01]
for a in alpha:
    lr = Lasso(alpha=a).fit(X_train, y_train)
    train = lr.score(X_train,y_train)
    test = lr.score(X_test,y_test)
    print(f'Alpha: {a}\nTrain: {train}\tTest: {test}')
    print(f'Number of features used: {np.sum(lr.coef_ != 0)}')
    train_acc.append(train)
    test_acc.append(test)

plt.plot(alpha, train_acc, label='Train')
plt.plot(alpha, test_acc, label='Test')
plt.legend()

# %%
# Combination of Ridge and Lasso --> Elastic net
# Lasso (uses L1 regularization) model on Boston Housing data
X, y = ld.load_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
lr = Lasso().fit(X_train, y_train)
print(f'Train score: {lr.score(X_train,y_train)}')
print(f'Test score: {lr.score(X_test,y_test)}')

# Finding optimal alpha for lasso regression
cols = ['Alpha', 'Ratio', 'Train', 'Test']
score_tracker = pd.DataFrame(columns=cols)
alpha = np.arange(0,1,0.01).tolist()
ratio = np.arange(0,1.1,0.1).tolist()
for r in ratio:
    for a in alpha:
        lr = Lasso(alpha=a).fit(X_train, y_train)
        train = lr.score(X_train,y_train)
        test = lr.score(X_test,y_test)
        #print(f'Alpha: {a}\t L1/L2: {r}\nTrain: {train}\tTest: {test}')
        temp_df = pd.DataFrame({'Alpha': [a], 'Ratio': [r],
                                 'Train': [train], 'Test': [test]})
        score_tracker = pd.concat([score_tracker, temp_df], axis=0)

    
#%%
# Plots
plt.plot(alpha, train_acc[-100:], label='Train')
plt.plot(alpha, test_acc[-100:], label='Test')
plt.legend()

# %%
fig, axes = plt.subplots(2,5, sharex=True, sharey=True)
i=0
for ax in (axes.flatten()):
    temp_df = score_tracker.iloc[i:i+100, :]
    ax.grid(True)
    ax.plot(alpha, temp_df.iloc[:, 2], marker='o', markersize=5, label='Train')
    ax.plot(alpha, temp_df.iloc[:, 3], marker='o', markersize=5, label='Test')
    ax.set_title(f'Ratio: {temp_df.Ratio.median()}')
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Score')
    ax.legend()
    i+=100

# %%
# Applying logistic regression on Forge data
'''
    Despite the name, Logistic regression is used for classification.
'''
X, y = ld.load_forge()
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
    clf = model.fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5,
    ax=ax, alpha=.7)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{}".format(clf.__class__.__name__))
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
axes[0].legend()

#%%
# Applying logreg to breast cancer
X, y, fn, tn = ld.load_cancer()
X_train, X_test, y_train, y_test = train_test_split(
X, y, stratify=y, random_state=60)
logreg = LogisticRegression().fit(X_train, y_train)
print(f'Train score: {logreg.score(X_train, y_train)}')
print(f'Test score: {logreg.score(X_test, y_test)}')

# %%
## Trying different values of C
C = np.arange(1,100,1).tolist()
train_score = []
test_score = []
for c in C:
    logreg = LogisticRegression(C=c).fit(X_train, y_train)
    train = logreg.score(X_train, y_train)
    test = logreg.score(X_test, y_test)
    print(f'C: {c}========================================')
    print(f'Train score: {train}')
    print(f'Test score: {test}')
    train_score.append(train)
    test_score.append(test)
plt.plot(C, train_score, label='Train')
plt.plot(C, test_score, label='Test')
plt.xlabel('C')
plt.ylabel('Score')
plt.legend()

# %%
# Making blobs and building a classifier
X, y = make_blobs(random_state=42)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(["Class 0", "Class 1", "Class 2"])
linear_svm = LinearSVC().fit(X, y)
print(f"Coefficient shape: {linear_svm.coef_.shape}")
print(f"Intercept shape: {linear_svm.intercept_.shape}")

mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
line = np.linspace(-15, 15)
for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_,
['b', 'r', 'g']):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(['Class 0', 'Class 1', 'Class 2', 'Line class 0', 'Line class 1',
'Line class 2'])

mglearn.plots.plot_2d_classification(linear_svm, X, fill=True, alpha=.7)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
line = np.linspace(-15, 15)
for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_,
['b', 'r', 'g']):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
plt.legend(['Class 0', 'Class 1', 'Class 2', 'Line class 0', 'Line class 1',
'Line class 2'])
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

# %%
'''
    The main parameter of linear models is the regularization parameter, called alpha inthe regression models and C in LinearSVC and LogisticRegression. Large values foralpha or small values for C mean simple models. In particular for the regression models,tuning these parameters is quite important. Usually C and alpha are searched foron a logarithmic scale. The other decision you have to make is whether you want touse L1 regularization or L2 regularization. If you assume that only a few of your featuresare actually important, you should use L1. Otherwise, you should default to L2.L1 can also be useful if interpretability of the model is important. As L1 will use onlya few features, it is easier to explain which features are important to the model, andwhat the effects of these features are.
'''
