import pandas as pd

# sample submissions
sample_output = pd.read_csv('sample_submission.csv', sep=",")
sample_output.head()

# data types for each col
print(sample_output.dtypes)

# Loading and exploring train data
# Column names on the train.csv file and corresponding data types:
from DataGen import DataGen
train_data_generator = DataGen(file_path='train.csv', chunk_size=10000000)
print(train_data_generator.col_names)
print('------------------------------------')

next_batch, end_of_file = train_data_generator.next_batch()

print('Duplicates in "time_to_failure"? {}'.format(
    not next_batch.loc[next_batch['time_to_failure'].duplicated()].empty))

next_batch.head()
# data types for each col
print(next_batch.dtypes)
print('------------------------------------')
print('Duplicates in "time_to_failure"? {}'.format(
    not next_batch.loc[next_batch['time_to_failure'].duplicated()].empty))

import matplotlib.pyplot as plt
# plot second batch of data and zoom

fig, axs = plt.subplots(2, 2, figsize=(16, 8))
next_batch.plot(x='time_to_failure', y='acoustic_data', ax=axs[0, 0])
next_batch[10000:11000].plot(x='time_to_failure', y='acoustic_data', ax=axs[0, 1])
next_batch[10000:10300].plot(x='time_to_failure', y='acoustic_data', ax=axs[1, 0])
next_batch[10000:10100].plot(x='time_to_failure', y='acoustic_data', ax=axs[1, 1])

if next_batch.empty:
    train_data_generator = DataGen(file_path='train.csv', chunk_size=10000000)
    next_batch, end_of_file = train_data_generator.next_batch()

maxx = next_batch.max(axis=0)
minn = next_batch.min(axis=0)
summ = next_batch.sum(axis=0)
max_acoustic = maxx[0]
sum_acoustic = summ[0]
min_acoustic = minn[0]
max_time = maxx[1]
sum_time = summ[1]
min_time = minn[1]
data_points = next_batch.shape[0]
while True:
    maxx = next_batch.max(axis=0)
    minn = next_batch.min(axis=0)
    if maxx[0] > 5000.0:
        next_batch.plot(x='time_to_failure', y='acoustic_data', label='anomalous large max acoustic spike')
    if maxx[0] > max_acoustic:
        max_acoustic = maxx[0]
    if minn[0] < min_acoustic:
        min_acoustic = minn[0]
    if maxx[1] > max_time:
        max_time = maxx[1]
    if minn[1] < min_time:
        min_time = minn[1]

    next_batch, end_of_file = train_data_generator.next_batch()
    if end_of_file:
        break
    data_points += next_batch.shape[0]
    summ = next_batch.sum(axis=0)
    sum_acoustic += summ[0]
    sum_time += summ[1]

stats = {
    'max_acoustic': max_acoustic, 'min_acoustic': min_acoustic, 'max_time': max_time, 'min_time': min_time,
    'avg_acoustic': sum_acoustic / data_points, 'ave_time': sum_time / data_points
}


if next_batch.empty:
    train_data_generator = DataGen(file_path='train.csv', chunk_size=10000000)
    next_batch, end_of_file = train_data_generator.next_batch()

import matplotlib.pyplot as plt
from scipy import signal
from math import pi, e
from DataGen import DataGen

fig, axs = plt.subplots(2, 2, figsize=(16, 4))
train_data_generator = DataGen(file_path='train.csv', chunk_size=10000000)
next_batch, end_of_file = train_data_generator.next_batch()
current_batch = next_batch['acoustic_data'][10000:11000]
next_batch[10000:11000].plot(x='time_to_failure', y='acoustic_data', ax=axs[0, 0])

sigma = [0.1, 0.5, 1]
t = M = [32, 48, 56] # windows sizes
w = 2*pi # Omega0 - base angular freq

sigma_term = e**(-(sigma[0]**2)/2)
morlet = signal.morlet(t[0], w=2*pi)
F_W = signal.fftconvolve(current_batch, morlet - sigma_term, mode='same')
axs[0, 1].plot(F_W, label="sigma = {}".format(sigma[0]))
axs[0, 1].legend()

sigma_term = e**(-(sigma[1]**2)/2)
morlet = signal.morlet(t[1], w=2*pi)
F_W = signal.fftconvolve(current_batch, morlet - sigma_term, mode='same')
axs[1, 0].plot(F_W, label="sigma = {}".format(sigma[1]))
axs[1, 0].legend()

sigma_term = e**(-(sigma[2]**2)/2)
morlet = signal.morlet(t[2], w=2*pi)
F_W = signal.fftconvolve(current_batch, morlet - sigma_term, mode='same')
axs[1, 1].plot(F_W, label="sigma = {}".format(sigma[2]))
axs[1, 1].legend()

import numpy as np
def preprocess(current_batch):
    out_batch = np.ndarray(shape=(3,), dtype=np.int16)
    for i in range(len((sigma))):
        sigma_term = e**(-(sigma[i]**2)/2)
        morlet = signal.morlet(t[i], w=2*pi)
        C_psi = 2 * pi * signal.correlate(np.fft.fft(current_batch), np.conj(np.fft.fft(current_batch)), mode='full').sum()
        F_W = signal.fftconvolve(current_batch, morlet - sigma_term, mode='same')
        F_W = np.dot(F_W[100:900], F_W[100:900]).real / (C_psi.real + 1)
        out_batch[i] = (F_W * 100).astype(np.int16)
    return out_batch


import tensorflow as tf
train_data_generator = DataGen(file_path='train.csv', chunk_size=10000000)
sum_time = 0
while True:
    next_batch, end_of_file = train_data_generator.next_batch()
    if end_of_file:
        break
    data_points = next_batch.shape[0]
    maxx = next_batch.max(axis=0)
    minn = next_batch.min(axis=0)
    summ = next_batch.sum(axis=0)
    max_time = maxx[1]
    sum_time += summ[1]
    min_time = minn[1]
    preprocess()
