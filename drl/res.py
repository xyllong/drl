import os
import d3rlpy
import pandas as pd
import matplotlib.pyplot as plt

logs = os.listdir('d3rlpy_logs')

exp_res = {'BCQ': {}, 'CQL': {}, 'IQL': {}}

dataset_name = 'halfcheetah-medium-replay-v0'
dataset, env = d3rlpy.datasets.get_dataset(dataset_name)
dataset_best_returns =  max([episode.compute_return() for episode in dataset.episodes])

for log_path in logs:
    exp, trim_pct = log_path.split('_')[0], log_path.split('_')[-3]
    trim_pct = int(trim_pct[-1]) / 10
    scores = pd.read_csv(f'./d3rlpy_logs/{log_path}/environment.csv', header=None)[2].values
    exp_res[exp][trim_pct] = scores.max()

df = pd.DataFrame(exp_res).sort_index()
print(df.head())
df.plot.line()
plt.hlines(dataset_best_returns, 0, 1, linestyles='--')
plt.savefig('test.png')