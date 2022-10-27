import pandas as pd
import sys
import os
import numpy as np
from collections import defaultdict 

paths = sys.argv[1:]

probe_results = defaultdict(list)
knn_results = defaultdict(list)
MAX_TASK = 19
for path in paths:
    if not os.path.isdir(path): continue
    for tsv in os.listdir(path):
        if 'stats_probe_train' not in tsv: continue
        full_path = os.path.join(path, tsv)
        frac = tsv.split('-')[1]
        table = pd.read_csv(full_path, sep='\t', index_col=0)
        res = table[table['task'] == max(table['task'])]  
        if max(table['task']) != MAX_TASK:
            continue
        probe_results[frac].append(max(res['probe_mean_acc']))
        knn_results[frac].append(max(res['knn_mean_acc']))

for (k, lis_v) in sorted(knn_results.items()):
    print(f"train data frac: {k}, mean knn_mean_acc: {np.mean(lis_v):.2f}, std: {np.std(lis_v):.2f}")

for (k, lis_v) in sorted(probe_results.items()):
    print(f"train data frac: {k}, mean probe_mean_acc: {np.mean(lis_v):.2f}, std: {np.std(lis_v):.2f}")