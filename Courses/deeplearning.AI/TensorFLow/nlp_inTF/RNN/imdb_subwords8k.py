'''
Developer: vkyprmr
Filename: imdb_subwords8k.py
Created on: 2020-09-09 at 11:53:22
'''
'''
Modified by: vkyprmr
Last modified on: 2020-09-09 at 14:59:45
'''

#%%
# Imports
from __future__ import absolute_import, division, print_function, unicode_literals

from datetime import datetime
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
%matplotlib qt

#%%
# Data
dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)    # 8k specifies the vocab size
train_dataset, test_dataset = dataset['train'], dataset['test']

# Tokenizer
tokenizer = info.features['text'].encoder
BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(train_dataset))
test_dataset = test_dataset.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(test_dataset))

#%%
# Building the model
embedding_dim = 64
vocab_size = tokenizer.vocab_size

model_name = f'emb-{vocab_size}{embedding_dim}_bdlstm-6432_64'

model = Sequential(layers=[
                            Embedding(vocab_size, embedding_dim),
                            Bidirectional(LSTM(64, return_sequences=True)),
                            Bidirectional(LSTM(32)),
                            Dense(64, activation='relu'),
                            Dense(1, activation='sigmoid')
                            ],
                    name=model_name
                            )
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

#%%
# Training
epochs = 15

log_dir = "logs\\fit\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir, histogram_freq=1, profile_batch=0)

history = model.fit(train_dataset, epochs=epochs, validation_data=test_dataset, verbose=2, callbacks=[tensorboard_callback])

#%%
# Visualizing Accuracy and Loss during training
def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric])
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])
    plt.show()

plot_graphs('accuracy')
plot_graph('loss')

