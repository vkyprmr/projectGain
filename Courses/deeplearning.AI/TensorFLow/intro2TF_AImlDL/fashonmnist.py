"""
Developer: vkyprmr
Filename: fashonmnist.py
Created on: 2020-09-04 at 19:59:45
"""
"""
Modified by: vkyprmr
Last modified on: 2020-09-09 at 14:51:24
"""

# Import
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras import datasets
from tensorflow.keras.callbacks import TensorBoard


# Loading and Preparing data
(X_train, y_train), (X_test, y_test) = datasets.fashion_mnist.load_data()
plt.imshow(X_train[0])
X_train, X_test = X_train/255.0, X_test/255.0


# Defining basic model
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

model.fit(X_train, y_train, epochs=10)


# Defining own callbacks
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>=0.99):
      print("\nReached 90% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()
model = Sequential(
                    [
                        Flatten(),
                        Dense(512, activation=tf.nn.relu),
                        Dense(10, activation=tf.nn.softmax)
                    ]
                   )
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

log_dir = "logs\\fit\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir, histogram_freq=1, profile_batch=0)

callbacks = [callbacks, tensorboard_callback]

model.fit(X_train, y_train, epochs=5, callbacks=[callbacks])
