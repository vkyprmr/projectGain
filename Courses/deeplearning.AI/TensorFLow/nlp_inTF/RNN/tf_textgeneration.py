"""
Developer: vkyprmr
Filename: tf_textgeneration.py
Created on: 2020-09-09 at 23:57:50
"""
"""
Modified by: vkyprmr
Last modified on: 2020-09-10 at 00:43:41
"""


# Imports
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
physical_devices = tf.config.experimental.list_physical_devices('GPU') 
for physical_device in physical_devices: 
    tf.config.experimental.set_memory_growth(physical_device, True)


# Data
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
# Read, then decode for py2 compat.
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# length of text is the number of characters in it
print (f'Length of text: {len(text)} characters')

# The unique characters in the file
vocab = sorted(set(text))
print (f'{len(vocab)} unique characters')

# Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])
print('{')
for char,_ in zip(char2idx, range(20)):
    print(f'  {repr(char):4s}: {char2idx[char]:3d},')
print('  ...\n}')


# Preperation for model and predictions
# The maximum length sentence we want for a single input in characters
seq_length = 100
examples_per_epoch = len(text)//(seq_length+1)

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

for i in char_dataset.take(5):
    print(idx2char[i.numpy()])

sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

### The batch method lets one easily convert these individual characters to sequences of the desired size.
for item in sequences.take(5):
    print(repr(''.join(idx2char[item.numpy()])))

### For each sequence, duplicate and shift it to form the input and target text by using the map method to apply a simple function to each batch
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

for input_example, target_example in  dataset.take(1):
    print ('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
    print ('Target data:', repr(''.join(idx2char[target_example.numpy()])))


# Building the model
# Batch size
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

vocab_size = len(vocab)
embedding_dim = 256

model_name = f'emb-{vocab_size}{embedding_dim}_gru-6432_64'

def build_model(vocab_size, embedding_dim, batch_size, model_name):
    model = Sequential(layers=[
                                Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
                                GRU(64, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
                                Dense(vocab_size)
                                ],
                        name=model_name
                        )
model = build_model(vocab_size, embedding_dim, BATCH_SIZE, model_name)                        
#optimizer = Adam(lr=0.001)                            
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

""" 
    Another way to define loss...
    def loss(labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
 """

model.summary()


# Checking the shape of the output
for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")


# Train the model
epochs = 15

log_dir = "logs\\fit\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir, histogram_freq=1, profile_batch=0)

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

history = model.fit(dataset, epochs=epochs, callbacks=[checkpoint_callback, tensorboard_callback])


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


# Predictions
""" 
    To keep this prediction step simple, use a batch size of 1.
    Because of the way the RNN state is passed from timestep to timestep, the model only accepts a fixed batch size once built.
    To run the model with a different batch_size, we need to rebuild the model and restore the weights from the checkpoint.
 """

tf.train.latest_checkpoint(checkpoint_dir)
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))
model.summary()

### Prediction function
def generate_text(model, start_string):
    # Evaluation step (generating text using the learned model)
    # Number of characters to generate
    num_generate = 1000

    # Converting our start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = 1.0

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        # We pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))

print(generate_text(model, start_string=u"ROMEO: "))


# Customized training
### R
# ead more on: https://www.tensorflow.org/tutorials/text/text_generation
model = build_model(vocab_size, embedding_dim, BATCH_SIZE, model_name)                        
optimizer = Adam()

@tf.function
def train_step(inp, target):
    with tf.GradientTape() as tape:
    predictions = model(inp)
    loss = tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(
            target, predictions, from_logits=True))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss

### Training step
EPOCHS = 10

for epoch in range(EPOCHS):
    start = time.time()

    # resetting the hidden state at the start of every epoch
    model.reset_states()

    for (batch_n, (inp, target)) in enumerate(dataset):
        loss = train_step(inp, target)

        if batch_n % 100 == 0:
            print(f'Epoch {epoch+1} Batch {batch_n} Loss {loss}')

    # saving (checkpoint) the model every 5 epochs
    if (epoch + 1) % 5 == 0:
        model.save_weights(checkpoint_prefix.format(epoch=epoch))

    print(f'Epoch: {epoch+1} Loss: {loss:.4f}')
    print(f'Time taken for 1 epoch: {time.time() - start} sec\n')

model.save_weights(checkpoint_prefix.format(epoch=epoch))
