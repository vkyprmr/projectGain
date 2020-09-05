'''
Developer: vkyprmr
Filename: Untitled-1
Created on: 2020-09-02 at 22:13:37
'''
'''
Modified by: vkyprmr
Last modified on: 2020-09-03 at 15:54:32
'''

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mglearn
from sklearn import datasets


# Generating a class to make it easier to perform various actions and no need to load data in every file


class LoadData:
    def __init__(self):
        self.available = ['Forge data',
                          'Wave data',
                          'Breast-Cancer data',
                          'Boston-Housing data']
        print('Available Data:')
        for l in zip(range(len(self.available)), self.available):
            print(f'{l[0]+1}. {l[1]}')
        print('For details on different methods please read loaddata.md in this folder')

    def load_forge(self, visualize=False):
        self.X, self.y = mglearn.datasets.make_forge()
        if visualize:
            mglearn.discrete_scatter(self.X[:,0], self.X[:,1], self.y)
            plt.legend(['Class 0', 'Class 1'])
            plt.xlabel('Feature A')
            plt.ylabel('Feature B')
        return self.X, self.y

    def load_wave(self, n_samples=40, visualize=False):
        self.X, self.y = mglearn.datasets.make_wave(n_samples)
        if visualize:
            plt.plot(X, y, 'o')
            plt.ylim(-3, 3)
            plt.xlabel("Feature")
            plt.ylabel("Target")
        return self.X, self.y

    def load_cancer(self):
        self.cancer_dct = datasets.load_breast_cancer()
        print(f'Data loaded as dictionary with keys: {self.cancer_dct.keys()}')
        self.X = self.cancer_dct['data']
        self.y = self.cancer_dct['target']
        self.feature_names = self.cancer_dct['feature_names']
        self.target_names = self.cancer_dct['target_names']
        print(f'Sample counts per class - {({n: v for n, v in zip(self.target_names, np.bincount(self.y))})}')
        print('Data loaded successfully')
        return self.X, self.y, self.feature_names, self.target_names

    def load_boston(self, extended=True):
        if extended:
            self.X, self.y = mglearn.datasets.load_extended_boston()
            return self.X, self.y
        else:
            self.boston_dct = datasets.load_boston()
            self.X = self.boston_dct['data']
            self.y = self.boston_dct['targer']
            self.feature_names = self.boston_dct['feature_names']
            return self.X, self.y, self.feature_names


