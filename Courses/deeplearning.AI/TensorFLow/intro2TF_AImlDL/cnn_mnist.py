'''
Developer: vkyprmr
Filename: cnn_mnist.py
Created on: 2020-09-04 at 21:26:57
'''
'''
Modified by: vkyprmr
Last modified on: 2020-09-09 at 14:51:36
'''

#%%
# Import
import numpy as np
import matplotlib.pyplot as plt
%matplotlib qt
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras import datasets
from tensorflow.keras.callbacks import TensorBoard

#%%
# Loading and Preparing data
(X_train, y_train), (X_test, y_test) = datasets.fashion_mnist.load_data()
plt.imshow(X_train[0])
X_train, X_test = X_train/255.0, X_test/255.0

# %%
# Defining basic model
###
##### Defining own callbacks


class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>=0.99):
      print("\nReached 99% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()

model = Sequential(
                    [
                        Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
                        MaxPooling2D(2, 2),
                        Conv2D(64, (3,3), activation='relu'),
                        MaxPooling2D(2,2),
                        Flatten(),
                        Dense(128, activation='relu'),
                        Dense(10, activation='softmax')
                    ]
                   )
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

batch_size, width, height, channels = -1, 28, 28, 1

log_dir = "logs\\fit\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir, histogram_freq=1, profile_batch=0)

callbacks = [callbacks, tensorboard_callback]

model.fit(X_train.reshape(batch_size, width, height, channels), y_train, epochs=10, callbacks=[callbacks])

#%%
# Check what each layer is doing
f, axarr = plt.subplots(3,4)
FIRST_IMAGE=0
SECOND_IMAGE=7
THIRD_IMAGE=26
CONVOLUTION_NUMBER = 1
layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)
for x in range(0,4):
  f1 = activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[0,x].imshow(f1[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[0,x].grid(False)
  f2 = activation_model.predict(test_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[1,x].imshow(f2[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[1,x].grid(False)
  f3 = activation_model.predict(test_images[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[2,x].imshow(f3[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[2,x].grid(False)

