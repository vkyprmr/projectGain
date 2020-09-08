'''
Developer: vkyprmr
Filename: imdb_embeddings_subwords.py
Created on: 2020-09-08 at 23:16:37
'''
'''
Modified by: vkyprmr
Last modified on: 2020-09-08 at 23:26:55
'''

#%%
# Imports
import io
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten, GlobalAveragePooling1D
from tensorflow.keras.optimizers import RMSprop, Adam

#%%
# Set parameters
""" 
    tf.enable_eager_excecution      # if using tensorflow 1.x
    config = tf.compat.v1.ConfigProto(device_count = {'GPU': 1})
    sess = tf.compat.v1.Session(config=config)
 """
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
physical_devices = tf.config.experimental.list_physical_devices('GPU') 
for physical_device in physical_devices: 
    tf.config.experimental.set_memory_growth(physical_device, True)

#%%
# Loading data from TFDS
imdb, info = tfds.load("imdb_reviews/subwords8k", with_info=True, as_supervised=True)

#%%
# Fetching data from the object
train_data, test_data = imdb['train'], imdb['test']

# %%
# Tokenizing data
vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type='post'
oov_tok = "<OOV>"

tokenizer = info.features['text'].encoder
print(tokenizer.subwords)

sample_string = 'TensorFlow, from basics to mastery'
tokenized_string = tokenizer.encode(sample_string)
print (f'Tokenized string is {tokenized_string}')
original_string = tokenizer.decode(tokenized_string)
print (f'The original string: {original_string}')
for ts in tokenized_string:
  print ('{} ----> {}'.format(ts, tokenizer.decode([ts])))

#%%
# Hyperparmeters
BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_dataset = train_data.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(train_dataset))
test_dataset = test_data.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(test_data))

#%%
# Building the model
embedding_dim = 64

model = Sequential(
                    [
                            Embedding(vocab_size, embedding_dim, input_length=max_length),
                            GlobalAveragePooling1D(),          # Using subwords tokenizer will cause tf to crash if Flatten, so use GlobalAveragePooling1D
                            Dense(6, activation='relu'),
                            Dense(1, activation='sigmoid')
                    ]
                    )
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

#%%
# Training
epochs = 10
model.fit(train_dataset, epochs=epochs, validation_data=test_dataset)

#%%
# Saving the weights for visualization
e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, vocab_size):
  word = tokenizer.decode([word_num])
  embeddings = weights[word_num]
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()

#%%
# Colab
try:
  from google.colab import files
except ImportError:
  pass
else:
  files.download('vecs.tsv')
  files.download('meta.tsv')
