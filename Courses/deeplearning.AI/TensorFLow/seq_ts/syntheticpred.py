"""
Developer: vkyprmr
Filename: syntheticpred.py
Created on: 2020-09-10 at 14:45:26
"""
"""
Modified by: vkyprmr
Last modified on: 2020-09-10 at 15:35:27
"""


# Imports
import numpy as np
import matplotlib.pyplot as plt

from pandas.plotting import autocorrelation_plot
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Data (synthetic)
def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

def trend(time, slope=0):
    return slope * time

def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.3,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.sin(3 * season_time))

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
amplitude = 40
slope = 0.05
noise_level = 5

# Create the series
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
# Update with noise
series += noise(time, noise_level, seed=42)

plot_series(time, series)


# Splitting data and visualizing
split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

plot_series(time_train, x_train)
plt.show()

plot_series(time_valid, x_valid)


# Naive Forecast
naive_forecast = series[split_time - 1:-1]

plot_series(time_valid, x_valid)
plot_series(time_valid, naive_forecast)

plot_series(time_valid, x_valid, start=0, end=125)
plot_series(time_valid, naive_forecast, start=1, end=151)

print(f'MSE: {mean_squared_error(x_valid, naive_forecast)}')
print(f'MAE: {mean_absolute_error(x_valid, naive_forecast)}')

### Moving Average
def moving_average_forecast(series, window_size):
    """Forecasts the mean of the last few values.
        If window_size=1, then this is equivalent to naive forecast"""
    forecast = []
    for time in range(len(series) - window_size):
        forecast.append(series[time:time + window_size].mean())
    return np.array(forecast)

moving_avg = moving_average_forecast(series, 30)[split_time - 30:]

plot_series(time_valid, x_valid)
plot_series(time_valid, moving_avg)

print(f'MSE: {mean_squared_error(x_valid, moving_avg)}')
print(f'MAE: {mean_absolute_error(x_valid, moving_avg)}')

### Removing trend and seasonality
diff_series = (series[365:] - series[:-365])
diff_time = time[365:]

plot_series(diff_time, diff_series)

##### MA
diff_moving_avg = moving_average_forecast(diff_series, 25)[split_time - 365 - 25:]

plot_series(time_valid, diff_series[split_time - 365:])
plot_series(time_valid, diff_moving_avg)

""" Getting actual predictions """
diff_moving_avg_plus_past = series[split_time - 365:-365] + diff_moving_avg

plot_series(time_train, x_train)
plot_series(time_valid, x_valid)
plot_series(time_valid, diff_moving_avg_plus_past)

print(f'MSE: {mean_squared_error(x_valid, diff_moving_avg_plus_past)}')
print(f'MAE: {mean_absolute_error(x_valid, diff_moving_avg_plus_past)}')

##### Removing past noise
diff_moving_avg_plus_smooth_past = moving_average_forecast(series[split_time - 370:-360], 10) + diff_moving_avg

plot_series(time_valid, x_valid)
plot_series(time_valid, diff_moving_avg_plus_smooth_past)

print(f'MSE: {mean_squared_error(x_valid, diff_moving_avg_plus_smooth_past)}')
print(f'MAE: {mean_absolute_error(x_valid, diff_moving_avg_plus_smooth_past)}')
