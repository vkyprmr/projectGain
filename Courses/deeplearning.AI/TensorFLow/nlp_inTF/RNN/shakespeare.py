'''
Developer: vkyprmr
Filename: shakespeare.py
Created on: 2020-09-10 at 00:48:50
'''
'''
Modified by: vkyprmr
Last modified on: 2020-09-11 at 18:25:57
'''

#%%
# Imports
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.utils as ku
from tensorflow.keras.regularizers import l2
import numpy as np 
from datetime import datetime
import matplotlib.pyplot as plt
%matplotlib qt

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
physical_devices = tf.config.experimental.list_physical_devices('GPU') 
for physical_device in physical_devices: 
    tf.config.experimental.set_memory_growth(physical_device, True)

#%%
# Data
filename = '../Data/Shakespeare/sonnets.txt'
data = open(filename).read()
corpus = data.lower().split("\n")

tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

# create input sequences using list of tokens
input_sequences = []
for line in corpus:
	token_list = tokenizer.texts_to_sequences([line])[0]
	for i in range(1, len(token_list)):
		n_gram_sequence = token_list[:i+1]
		input_sequences.append(n_gram_sequence)

# pad sequences 
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# create predictors and label
predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
label = ku.to_categorical(label, num_classes=total_words)

#%%
# Building the model
embedding_dim = 64
vocab_size = total_words

model_name = f'emb-{vocab_size}{embedding_dim}_bdlstm-6432_64'

model = Sequential(layers=[
                            Embedding(vocab_size, embedding_dim),
                            Bidirectional(LSTM(64, return_sequences=True))
                            Dropout(0.2)
                            Bidirectional(LSTM(32)),
                            Dense(total_words//2, kernel_regularizer=l2(0.01), activation='relu'),
                            Dense(total_words, activation='softmax')
                            ],
                    name=model_name
                            )
optimizer = Adam(lr=0.001)                            
model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
model.summary()

#%%
# Training
epochs = 15

log_dir = "logs\\fit\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir, histogram_freq=1, profile_batch=0)

history = model.fit(predictors, label, epochs=epochs, verbose=1, callbacks=[tensorboard_callback])

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
