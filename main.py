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

max_acoustic = next_batch.max(axis=0)[0]
min_acoustic = next_batch.min(axis=0)[0]
max_time = next_batch.max(axis=0)[1]
min_time = next_batch.min(axis=0)[1]
while not end_of_file:
    next_batch, end_of_file = train_data_generator.next_batch()
    temp = next_batch.max(axis=0)[0]
    if temp > max_acoustic:
        max_acoustic = temp
    temp = next_batch.min(axis=0)[0]
    if temp < min_acoustic:
        min_acoustic = temp
    temp = next_batch.max(axis=0)[1]
    if temp > max_time:
        max_time = temp
    temp = next_batch.min(axis=0)[1]
    if temp > min_time:
        min_time = temp

