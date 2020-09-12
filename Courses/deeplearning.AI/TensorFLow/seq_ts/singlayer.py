'''
Developer: vkyprmr
Filename: singlayer.py
Created on: 2020-09-10 at 16:07:04
'''
'''
Modified by: vkyprmr
Last modified on: 2020-09-12 at 01:31:25
'''

#%%
# Imports
import numpy as np
import matplotlib.pyplot as plt
%matplotlib qt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import mean_absolute_error

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
physical_devices = tf.config.experimental.list_physical_devices('GPU') 
for physical_device in physical_devices: 
    tf.config.experimental.set_memory_growth(physical_device, True)

#%%
# Data
def plot_series(time, series, format="-", start=0, end=None, label=None):
    plt.plot(time[start:end], series[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    if label:
        plt.legend(fontsize=14)
    plt.grid(True)

### Adding trend to a series
def trend(time, slope=0):
    return slope * time

### Adding seasonality
def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.3,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.sin(3 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

"""
    Time series with seasonality and trend
"""
time = np.arange(4 * 365 + 1)
baseline = 10
slope = 0.1
amplitude = 40
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)

### Noise (sounds more practical)
def white_noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

noise_level = 5
noise = white_noise(time, noise_level, seed=42)

"""
    Adding noise to the series
 """
series += noise

plot_series(time, series)

### Try to predict it
split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

#%%
# Fixed parameters
window_size = 15
batch_size = 32
shuffle_buffer_size = 1000

#%%
# Preparing data (features and labels with TF)
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset

dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

#%%
# Building the model
l0 = tf.keras.layers.Dense(1, input_shape=[window_size])
model = tf.keras.models.Sequential([l0])
model.compile(loss="mse", optimizer=SGD(lr=1e-6, momentum=0.9))

#%%
# Training
model.fit(dataset,epochs=100,verbose=1)
print(f'Layer weights: {l0.get_weights()}')

#%%
# Predictions
def predict(series):
    forecast=[]
    for time in range(len(series) - window_size):
        pred = model.predict(series[time:time + window_size][np.newaxis])
        forecast.append(pred)
        print(f'Epoch: {time}\nActual: {series[time]}\tPredicted: {pred}')

    forecast = forecast[split_time-window_size:]
    results = np.array(forecast)[:, 0, 0]
    return results

results = predict(series)

plot_series(time_train, x_train)
plot_series(time_valid, x_valid)
plot_series(time_valid, results)

print(f'MAE: {mean_absolute_error(x_valid, results).numpy()}')

# %%
