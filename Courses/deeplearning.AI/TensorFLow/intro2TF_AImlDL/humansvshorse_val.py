"""
Developer: vkyprmr
Filename: humansvshorse_val.py
Created on: 2020-09-07 at 15:27:52
"""
"""
Modified by: vkyprmr
Last modified on: 2020-09-09 at 14:51:06
"""

# Imports
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import TensorBoard


""" 
    import zipfile

    local_zip = '/tmp/horse-or-human.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('/tmp/horse-or-human')
    zip_ref.close()
 """


# Data
# Directory with our training pictures
train_horse_dir = os.path.join('Data/Training/horses')
train_human_dir = os.path.join('Data/Training/humans')

# Directory with our validation pictures
val_horse_dir = os.path.join('Data/Validation/horses')
val_human_dir = os.path.join('Data/Validation/humans')

train_horse_names = os.listdir(train_horse_dir)
train_human_names = os.listdir(train_human_dir)
val_horse_names = os.listdir(val_horse_dir)
val_human_names = os.listdir(val_human_dir)

# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 4
ncols = 4

# Index for iterating over images
pic_index = 0


# Set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
next_horse_pix = [os.path.join(val_horse_dir, fname) 
                for fname in val_horse_names[pic_index-8:pic_index]]
next_human_pix = [os.path.join(val_human_dir, fname) 
                for fname in val_human_names[pic_index-8:pic_index]]

for i, img_path in enumerate(next_horse_pix+next_human_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()


# Building a model
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
                        Dense(1, activation='sigmoid')
                    ]
                    )
model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])
model.summary()


# Building Image Generator
train_datagen = ImageDataGenerator(rescale=1/255)
""" 
                                    rescale=1./255,
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    fill_mode='nearest'
 """
train_generator = train_datagen.flow_from_directory('Data/Training/',
                                                    target_size=(300, 300),
                                                    batch_size=128,
                                                    class_mode='binary'
                                                    )
val_datagen = ImageDataGenerator(rescale=1/255)
""" 
                                    rescale=1./255,
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    fill_mode='nearest'
 """
val_generator = val_datagen.flow_from_directory('Data/Validation/',
                                                    target_size=(300, 300),
                                                    batch_size=128,
                                                    class_mode='binary'
                                                    )


# Training
log_dir = "logs\\fit\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir, histogram_freq=1, profile_batch=0)

callbacks =[tensorboard_callback]

history = model.fit_generator(train_generator, steps_per_epoch=8,
                              validation_data=val_generator, validation_steps=8,
                              epochs=15, verbose=1, callbacks=callbacks)


# Prediction
path = ''

img = image.load_img(path, target_size=(300, 300))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict(images, batch_size=10)
print(classes[0])

if classes[0]>0.5:
    print(fn + " is a human")
else:
    print(fn + " is a horse")



# Visualization of each Conv. layer
import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Let's define a new Model that will take an image as input, and will output
# intermediate representations for all layers in the previous model after
# the first.
successive_outputs = [layer.output for layer in model.layers[1:]]
#visualization_model = Model(img_input, successive_outputs)
visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)
# Let's prepare a random input image from the training set.
horse_img_files = [os.path.join(train_horse_dir, f) for f in train_horse_names]
human_img_files = [os.path.join(train_human_dir, f) for f in train_human_names]
img_path = random.choice(horse_img_files + human_img_files)

img = load_img(img_path, target_size=(300, 300))  # this is a PIL image
x = img_to_array(img)  # Numpy array with shape (150, 150, 3)
x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)

# Rescale by 1/255
x /= 255

# Let's run our image through our network, thus obtaining all
# intermediate representations for this image.
successive_feature_maps = visualization_model.predict(x)

# These are the names of the layers, so can have them as part of our plot
layer_names = [layer.name for layer in model.layers[1:]]

# Now let's display our representations
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
  if len(feature_map.shape) == 4:
    # Just do this for the conv / maxpool layers, not the fully-connected layers
    n_features = feature_map.shape[-1]  # number of features in feature map
    # The feature map has shape (1, size, size, n_features)
    size = feature_map.shape[1]
    # We will tile our images in this matrix
    display_grid = np.zeros((size, size * n_features))
    for i in range(n_features):
      # Postprocess the feature to make it visually palatable
      x = feature_map[0, :, :, i]
      x -= x.mean()
      x /= x.std()
      x *= 64
      x += 128
      x = np.clip(x, 0, 255).astype('uint8')
      # We'll tile each filter into this big horizontal grid
      display_grid[:, i * size : (i + 1) * size] = x
    # Display the grid
    scale = 20. / n_features
    plt.figure(figsize=(scale * n_features, scale))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
 
