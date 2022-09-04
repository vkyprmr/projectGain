"""
Developer: vkyprmr
Filename: syntheticdata.py
Created on: 2020-09-10 at 13:45:57
"""
"""
Modified by: vkyprmr
Last modified on: 2020-09-10 at 14:45:38
"""


# Imports
import numpy as np
import matplotlib.pyplot as plt

from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA


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

"""
    Time series with trend
"""
time = np.arange(4 * 365 + 1)
baseline = 10
series = trend(time, 0.1)

plot_series(time, series)

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
    Time series with seasonality
"""
baseline = 10
amplitude = 40
series = seasonality(time, period=365, amplitude=amplitude)

plot_series(time, series)

"""
    Time series with seasonality and trend
"""
slope = 0.1
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)

plot_series(time, series)

### Noise (sounds more practical)
def white_noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

noise_level = 5
noise = white_noise(time, noise_level, seed=42)

plot_series(time, noise)

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

### Auto correlation
def autocorrelation(time, amplitude, seed=None, type=0):
    if type==0:
        rnd = np.random.RandomState(seed)
        φ1 = 0.5
        φ2 = -0.1
        ar = rnd.randn(len(time) + 50)
        ar[:50] = 100
        for step in range(50, len(time) + 50):
            ar[step] += φ1 * ar[step - 50]
            ar[step] += φ2 * ar[step - 33]
        return ar[50:] * amplitude
    else:
        rnd = np.random.RandomState(seed)
        φ = 0.8
        ar = rnd.randn(len(time) + 1)
        for step in range(1, len(time) + 1):
            ar[step] += φ * ar[step - 1]
        return ar[1:] * amplitude

series = autocorrelation(time, 10, seed=42, type=1)
plot_series(time[:200], series[:200])

### Trend with Auto correlation
series = autocorrelation(time, 10, seed=42, type=0) + trend(time, 2)
plot_series(time[:200], series[:200])

### Trend with Auto correlation and Seasonality
series = autocorrelation(time, 10, seed=42) + seasonality(time, period=50, amplitude=150) + trend(time, 2)
plot_series(time[:200], series[:200])

### Combining everything to look more like practical data
series = autocorrelation(time, 5, seed=42, type=1) + seasonality(time, period=50, amplitude=150) + trend(time, 2)
series2 = autocorrelation(time, 5, seed=42, type=1) + seasonality(time, period=50, amplitude=2) + trend(time, -0.25) + 1000
series[200:] = series2[200:]
series += white_noise(time, 5)
plot_series(time[:], series[:])


### Impulses
def impulses(time, num_impulses, amplitude=1, seed=None):
    rnd = np.random.RandomState(seed)
    impulse_indices = rnd.randint(len(time), size=10)
    series = np.zeros(len(time))
    for index in impulse_indices:
        series[index] += rnd.rand() * amplitude
    return series

series = impulses(time, 15, seed=42)
plot_series(time, series)

### ACR
def autocorrelation(source, φs):
    ar = source.copy()
    max_lag = len(φs)
    for step, value in enumerate(source):
        for lag, φ in φs.items():
            if step - lag > 0:
              ar[step] += φ * ar[step - lag]
    return ar

signal = impulses(time, 10, seed=42)
series = autocorrelation(signal, {1: 0.99})
plot_series(time, series)
plt.plot(time, signal, "k-")

signal = impulses(time, 10, seed=42)
series = autocorrelation(signal, {1: 0.70, 50: 0.2})
plot_series(time, series)
plt.plot(time, signal, "k-")

series_diff1 = series[1:] - series[:-1]
plot_series(time[1:], series_diff1)


### Pandas Auto correlation
autocorrelation_plot(series)

### Predictions using ARIMA
model = ARIMA(series, order=(5, 1, 0))
model_fit = model.fit(disp=0)
print(model_fit.summary())

