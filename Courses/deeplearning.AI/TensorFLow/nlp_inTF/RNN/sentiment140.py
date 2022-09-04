"""
Developer: vkyprmr
Filename: sentiment140.py
Created on: 2020-09-09 at 22:16:56
"""
"""
Modified by: vkyprmr
Last modified on: 2020-09-09 at 22:49:06
"""


# Imports
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
from datetime import datetime
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
physical_devices = tf.config.experimental.list_physical_devices('GPU') 
for physical_device in physical_devices: 
    tf.config.experimental.set_memory_growth(physical_device, True)


# Data
def read_data(filename):
    df = pd.read_csv(filename, encoding='latin', header=None)
    sentences = df.iloc[:,5]
    labels = df.iloc[:,0]
    labels[labels==4] = 1
    return np.array(sentences), np.array(labels)

filename = '../Data/Sentiment140/training.csv'

sentences, labels = read_data(filename)

### Parameters
max_length = 16
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size= 160000
test_portion=.2


# Tokenizing words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index
vocab_size=len(word_index)

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, maxlen=max_length, padding = padding_type, truncating=trunc_type)

split = int(test_portion * training_size)

test_sequences = padded[0:split]
training_sequences = padded[split:training_size]
test_labels = np.array(labels[0:split])
training_labels = np.array(labels[split:training_size])


# Loading pretrained model weights (embeddings)
"""
    The embeddings used for transfer learning are from the GloVe,
    also known as Global Vectors for Word Representation,
    available at: https://nlp.stanford.edu/projects/glove/
 """
embedding2_file = '../Data/Sentiment140/embeddings/glove.6B.100d.txt'
embeddings_index = {};
with open(embedding2_file, encoding='utf8') as f:
    for line in f:
        values = line.split();
        word = values[0];
        coefs = np.asarray(values[1:], dtype='float32');
        embeddings_index[word] = coefs;

embeddings_matrix = np.zeros((vocab_size+1, embedding_dim));
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word);
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector;

### print(len(embeddings_matrix))


# Building the model
embedding_dim = 100
vocab_size = len(embeddings_matrix)+1

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


# Training
epochs = 15

log_dir = "logs\\fit\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir, histogram_freq=1, profile_batch=0)

history = model.fit(training_sequences, training_labels, epochs=epochs, validation_data=(test_sequences, test_labels), verbose=1, callbacks=[tensorboard_callback])


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
