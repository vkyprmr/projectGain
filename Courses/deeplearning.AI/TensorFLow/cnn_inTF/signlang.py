'''
Developer: vkyprmr
Filename: signlang.py
Created on: 2020-09-08 at 15:26:01
'''
'''
Modified by: vkyprmr
Last modified on: 2020-09-08 at 15:26:02
'''

#%%
# Imports
import os
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

%matplotlib qt

#%%
# Data
def read_data(filename):
    """ 
        Reads the csv file using pandas and returns images and labels as a tuple
     """
    df = pd.read_csv(filename, delimiter=',')
    labels = np.array(df.iloc[:,0])
    images = df.iloc[:,1:]
    images = images.values.reshape(-1,28,28)


    return images, labels

train_file = 'Data/Sign-Language/train.csv'
test_file = 'Data/Sign-Language/test.csv'

train_images, train_labels = read_data(train_file)
test_images, test_labels = read_data(test_file)

print(f'Training input shape: {train_images.shape}, Labels: {train_labels.shape}')
print(f'Testing input shape: {test_images.shape}, Labels: {test_labels.shape}')

# %%
# Reshaping data to pass to CNN
train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)

# Generator
train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
train_generator = train_datagen.flow(train_images, train_labels,
                                     batch_size = 32)

validation_datagen = ImageDataGenerator(rescale = 1./255.)
validation_generator = validation_datagen.flow(test_images, test_labels,
                                     batch_size = 32)

#%%
# Building the MODEL
model = Sequential(
                    [
                        Conv2D(16, (3,3), activation='relu', input_shape=(300,300,3)),
                        MaxPooling2D(2,2),
                        Conv2D(32, (3,3), activation='relu'),
                        MaxPooling2D(2,2),
                        Conv2D(64, (3,3), activation='relu'),
                        MaxPooling2D(2,2),
                        Flatten(),
                        Dense(512, activation='relu'),
                        Dense(25, activation='softmax')
                    ]
                    )
model.compile(loss='sparse_categorical_crossentropy', optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])
model.summary()

#%%
# Training
history = model.fit_generator(train_generator, steps_per_epoch=len(train_images)//32,
                              epochs=15, verbose=1, validation_data=validation_generator,
                              validation_steps=len(test_images)//32)

model.evaluate(test_images, test_labels, batch_size=32, verbose=1)

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


