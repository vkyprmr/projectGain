'''
Developer: vkyprmr
Filename: digitsmnist.py
Created on: 2020-09-04 at 20:37:57
'''
'''
Modified by: vkyprmr
Last modified on: 2020-09-04 at 21:26:31
'''

#%%
# Imports
import numpy as np
import matplotlib.pyplot as plt
%matplotlib qt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras import datasets

#%%
# Loading and Preparing data
(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()
plt.imshow(X_train[0])
X_train, X_test = X_train/255.0, X_test/255.0

# %%
# Defining basic model
""" 
model = Sequential(
                    [
                        Flatten(),
                        Dense(128, activation='relu'),
                        Dense(10, activation='softmax')
                    ]
                   )
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

%timeit model.fit(X_train, y_train, epochs=10)
 """

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>=0.99):
      print("\nReached 90% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()
model = Sequential(
                    [
                        Flatten(),
                        Dense(512, activation='relu'),
                        Dense(10, activation='softmax')
                    ]
                   )
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(X_train, y_train, epochs=10, callbacks=[callbacks])
