'''
Developer: vkyprmr
Filename: transfer_learning.py
Created on: 2020-09-07 at 23:21:51
'''
'''
Modified by: vkyprmr
Last modified on: 2020-09-09 at 14:52:57
'''

#%%
# Imports
import os
from datetime import datetime
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard

%matplotlib qt

#%%
# Pre-Trained model
local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(input_shape = (150, 150, 3), 
                                include_top = False, 
                                weights = None)

pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers:
  layer.trainable = False
  
pre_trained_model.summary()
""" 
last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output
 """

#%%
# Callback
# Define a Callback class that stops training once accuracy reaches 97.0%
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.97):
      print("\nReached 97.0% accuracy so cancelling training!")
      self.model.stop_training = True

#%%
# Custom layers
# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024)(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)                  
# Add a final sigmoid layer for classification
x = layers.Dense(1, activation='sigmoid')(x)        # For multiclass - Number of classes, activation='softmax'    

model = Model(pre_trained_model.input, x) 

model.compile(optimizer = RMSprop(lr=0.0001), 
              loss = 'binary_crossentropy',     # For multiclass classification: categorical_crossentropy/sparse_categorical_crossentropy
              metrics = ['accuracy'])

model.summary()

#%%
# Define our example directories and files
""" 
    train_dir = '/tmp/training'
    validation_dir = '/tmp/validation'

    train_horses_dir = os.path.join(train_dir, 'horses')
    train_humans_dir = os.path.join(train_dir, 'humans')
    validation_horses_dir = os.path.join(validation_dir, 'horses')
    validation_humans_dir = os.path.join(validation_dir, 'humans')

    train_horses_fnames = os.listdir(train_horses_dir)
    train_humans_fnames = os.listdir(train_humans_dir)
    validation_horses_fnames = os.listdir(validation_horses_dir)
    validation_humans_fnames = os.listdir(validation_humans_dir)

    print(len(train_horses_fnames))
    print(len(train_humans_fnames))
    print(len(validation_horses_fnames))
    print(len(validation_humans_fnames))
 """
 
#%%
# Data Generator
# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale = 1.0/255.)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size = 20,
                                                    class_mode = 'binary', 
                                                    target_size = (150, 150))     

# Flow validation images in batches of 20 using test_datagen generator
validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                          batch_size  = 20,
                                                          class_mode  = 'binary', 
                                                          target_size = (150, 150))

#%%
# Training
callbacks = myCallback()

log_dir = "logs\\fit\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir, histogram_freq=1, profile_batch=0)

callbacks = [callbacks, tensorboard_callback]

history = model.fit_generator(train_generator,
                              validation_data = validation_generator,
                              steps_per_epoch = 100,
                              epochs = 10,
                              validation_steps = 50,
                              verbose = 1)

#%%
# Metrics
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()
