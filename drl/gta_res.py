import os
import pandas as pd
import matplotlib.pyplot as plt

logs = os.listdir('d3rlpy_logs')

exp_res = {'BCQ': {'w': [], 'nw': []}, 'CQL': {'w': [], 'nw': []}, 'IQL': {'w': [], 'nw': []}}
dataset_name = 'halfcheetah-medium-replay-v0'

for log_path in logs:
    config_fields = log_path.split('_')
    exp = config_fields[0]
    weighted = 'w' if 'weighted' in config_fields else 'nw'
    scores = pd.read_csv(f'./d3rlpy_logs/{log_path}/environment.csv', header=None)[2].values
    exp_res[exp][weighted].append(scores.max())

res = {'exp': [], 'w': [], 'nw': []}
for e, r in exp_res.items():
    for r_w, r_nw in zip(r['w'], r['nw']):
        res['exp'].append(e)
        res['w'].append(r_w)
        res['nw'].append(r_nw)
df = pd.DataFrame(res)
gp = df.groupby('exp')
means = gp.mean()
stds = gp.std()
means.plot.bar(yerr=stds)
plt.savefig('test.png')