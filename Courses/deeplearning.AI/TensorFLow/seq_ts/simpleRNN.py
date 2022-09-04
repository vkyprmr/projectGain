"""
Developer: vkyprmr
Filename: simpleRNN.py
Created on: 2020-09-11 at 20:16:10
"""
"""
Modified by: vkyprmr
Last modified on: 2020-09-11 at 22:56:15
"""


# Imports
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Lambda, SimpleRNN
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import mean_absolute_error
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)


# Data
def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

def trend(time, slope=0):
    return slope * time

def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

time = np.arange(4 * 365 + 1, dtype="float32")
baseline = 10
series = trend(time, 0.1)
baseline = 10
amplitude = 20
slope = 0.09
noise_level = 5

# Create the series
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
# Update with noise
series += noise(time, noise_level, seed=42)

split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

window_size = 20
batch_size = 32
shuffle_buffer_size = 1000


# Preaparing Data
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


# Learning Rate Scheduler
dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

model_name = f'seq_rnn_lrsch-1010'

model = Sequential(layers=[
                            Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]),
                            SimpleRNN(40, return_sequences=True),
                            SimpleRNN(40),
                            Dense(1),
                            Lambda(lambda x: x * 100.0)
                            ],
                    name=model_name
                            )

optimizer = SGD(lr=1e-8, momentum=0.9)
model.compile(loss="huber", optimizer=optimizer, metrics=['mae'])


# Training
epochs = 100
lr_schedule = LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 20))
history_lrs = model.fit(dataset, epochs=epochs, callbacks=[lr_schedule, tensorboard_callback], verbose=1)


# Visualizing Learning Rates
plt.semilogx(history_lrs.history['lr'], history_lrs.history["loss"])
plt.axis([1e-8, 1e-3, 0, 300])


# Repeat with the minimum learning rate

# Learning Rate Scheduler
dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

model_name = f'seq_rnn-1010'

model = Sequential(layers=[
                            Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]),
                            SimpleRNN(40, return_sequences=True),
                            SimpleRNN(40),
                            Dense(1),
                            Lambda(lambda x: x * 100.0)
                            ],
                    name=model_name
                            )

optimizer = SGD(lr=1e-8, momentum=0.9)
model.compile(loss="huber", optimizer=optimizer, metrics=['mae'])


# Training
epochs = 100
history_lrs = model.fit(dataset, epochs=epochs, callbacks=[tensorboard_callback], verbose=1)


# Predictions
forecast=[]
for time in range(len(series) - window_size):
  forecast.append(model.predict(series[time:time + window_size][np.newaxis]))

forecast = forecast[split_time-window_size:]
results = np.array(forecast)[:, 0, 0]

plot_series(time_valid, x_valid)
plot_series(time_valid, results)


# Visualization
plt.plot(history_lrs.history['loss'], label='Loss')
plt.pot(history_lrs.history['mae'], label='MAE')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss & MAE')
