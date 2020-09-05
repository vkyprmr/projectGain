'''
Developer: vkyprmr
Filename: knn.py
Created on: 2020-09-03 at 16:46:57
'''
'''
Modified by: vkyprmr
Last modified on: 2020-09-03 at 18:52:19
'''

#%%
# Imports
from loaddata import LoadData
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import mglearn
import numpy as np
import matplotlib.pyplot as plt
%matplotlib qt

#%%
# Data
ld = LoadData()

X_forge, y_forge = ld.load_forge(visualize=False)
X_forge_train, X_forge_test, y_forge_train, y_forge_test = train_test_split(X_forge, y_forge, random_state=0)

#%%
# Classifier
knc = KNeighborsClassifier(n_neighbors=2)
knc.fit(X_forge_train, y_forge_train)
preds = knc.predict(X_forge_test)

print(f'Score: {knc.score(X_forge_test, y_forge_test)}')

# %%
# Finding optimal 'n_neighbors' by comparing train and test scores/acc
train_acc = []
test_acc = []
K = range(1,10)
for k in K:
    knc = KNeighborsClassifier(n_neighbors=k)
    knc.fit(X_forge_train, y_forge_train)
    train = knc.score(X_forge_train, y_forge_train)
    train_acc.append(train)
    test = knc.score(X_forge_test, y_forge_test)
    test_acc.append(test)

# Plot the elbow
plt.plot(K, train_acc, 'bx-')
plt.plot(K, test_acc)
plt.xlabel('k')
plt.ylabel('Acc')
plt.title('Train acc vs. Test acc')
plt.show()

# %%
# Visualizing different possibilities
fig, axes = plt.subplots(2,3)
n_neighbors = [1,3,9,12,15,18]
for n_neighbors, ax in zip(n_neighbors, axes.flatten()):
    knc = KNeighborsClassifier(n_neighbors)
    knc.fit(X_forge, y_forge)
    mglearn.plots.plot_2d_separator(knc, X_forge, fill=True, eps=0.5, ax=ax, alpha=0.3)
    mglearn.discrete_scatter(X_forge[:,0], X_forge[:,1], y_forge, ax=ax)
    ax.set_title(f'{n_neighbors} neighbor(s)')
    ax.set_xlabel('Feature A')
    ax.set_ylabel('Feature B')
axes[0][0].legend()

'''
    As you can see on the left in the figure, using a single neighbor results in a decision boundary that follows the training data closely. Considering more and more neighbors leads to a smoother decision boundary. A smoother boundary corresponds to a simpler model. In other words, using few neighbors corresponds to high model complexity (as shown on the right side of Figure 2-1), and using many neighbors corresponds to low model complexity (as shown on the left side of Figure 2-1). If you consider the extreme case where the number of neighbors is the number of all data points in the training set, each test point would have exactly the same neighbors (all training points) and all predictions would be the same: the class that is most frequent in the training set.
'''

# %%
# Cancer data
X_cancer, y_cancer, fn_cancer, tn_cancer = ld.load_cancer()
X_cancer_train, X_cancer_test, y_cancer_train, y_cancer_test = train_test_split(X_cancer, y_cancer, stratify=y_cancer, random_state=42)

# Finding optimal 'n_neighbors' by comparing train and test scores/acc
train_acc = []
test_acc = []
K = range(1,15)
for k in K:
    knc = KNeighborsClassifier(n_neighbors=k)
    knc.fit(X_cancer_train, y_cancer_train)
    train = knc.score(X_cancer_train, y_cancer_train)
    train_acc.append(train)
    test = knc.score(X_cancer_test, y_cancer_test)
    test_acc.append(test)

# Plot the elbow
plt.plot(K, train_acc, label='Train')
plt.plot(K, test_acc, label='Test')
plt.xlabel('k')
plt.ylabel('Acc')
plt.title('Train acc vs. Test acc')
plt.legend()
plt.show()

# %%
# Making regression model for wave dataset
X_wave, y_wave = ld.load_wave(visualize=False)
X_wave_train, X_wave_test, y_wave_train, y_wave_test = train_test_split(X_wave, y_wave, random_state=0)
knr = KNeighborsRegressor(n_neighbors=3)
knr.fit(X_wave_train, y_wave_train)
print(f'R^2 score: {knr.score(X_wave_test, y_wave_test)}')

# %%
# Finding optimal 'n_neighbors' by comparing train and test scores/acc
fig, axes = plt.subplots(1, 3)
# create 1,000 data points, evenly spaced between -3 and 3
line = np.linspace(-3, 3, 1000).reshape(-1, 1)
for n_neighbors, ax in zip([1, 3, 9], axes):
    # make predictions using 1, 3, or 9 neighbors
    reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(X_wave_train, y_wave_train)
    ax.plot(line, reg.predict(line))
    ax.plot(X_wave_train, y_wave_train, '^', c=mglearn.cm2(0), markersize=4)
    ax.plot(X_wave_test, y_wave_test, 'v', c=mglearn.cm2(1), markersize=4)
    ax.set_title(
    "{} neighbor(s)\n train score: {:.2f} test score: {:.2f}".format(
    n_neighbors, reg.score(X_wave_train, y_wave_train),
    reg.score(X_wave_test, y_wave_test)))
    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")
axes[0].legend(["Model predictions", "Training data/target",
"Test data/target"], loc="best")

# %%
