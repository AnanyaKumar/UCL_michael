from torch import Tensor
from collections import OrderedDict
import os
from .plotter import Plotter
from collections import defaultdict
import pandas as pd
import shutil

class Logger(object):
    def __init__(self, log_dir, matplotlib=True):
        
        self.reset(log_dir, matplotlib)

    def reset(self, log_dir=None, tensorboard=True, matplotlib=True):

        if log_dir is not None: self.log_dir=log_dir 
        self.plotter = Plotter() if matplotlib else None
        self.counter = OrderedDict()

    def update_scalers(self, ordered_dict):

        for key, value in ordered_dict.items():
            if isinstance(value, Tensor):
                ordered_dict[key] = value.item()
            if self.counter.get(key) is None:
                self.counter[key] = 1
            else:
                self.counter[key] += 1

        if self.plotter: 
            self.plotter.update(ordered_dict)
            self.plotter.save(os.path.join(self.log_dir, 'plotter.svg'))


    @staticmethod
    def accumulate_batch_tuples(v):
        metric_sum = defaultdict(int)
        metric_counts = defaultdict(int)
        assert isinstance(v[0], tuple)
        res = []
        for (epoch_idx, value) in v:
            metric_sum[epoch_idx] += value
            metric_counts[epoch_idx] += 1
            res.append(metric_sum[epoch_idx] / metric_counts[epoch_idx])
        return res

    def process_stats(self, train_stats, test_stats=None):
        # Following baseline_train.py in https://github.com/AnanyaKumar/transfer_learning
        # train_stats and test_stats are dictionaries mapping metrics to either:
        # List of (epoch idx, value) for all epochs (e.g. loss, acc)
        # List of values, one for each epoch

        for (k, v) in train_stats.items():
            if len(v) > 1:
                v = self.accumulate_batch_tuples(v)
            train_stats[k] = v[0] if type(v[0]) == str else round(v[0], 2)

        if test_stats:                            
            for (k, v) in test_stats.items():
                if len(v) > 1:
                    v = self.accumulate_batch_tuples(v)
                test_stats[k] = v[0] if type(v[0]) == str else round(v[0], 2)
                

    def write_tsv(self, train_metrics, test_metrics=None, file="stats.tsv"):        
        train_df = pd.DataFrame(train_metrics)  
        num_epochs = max(train_df['epoch']) + 1      
        if test_metrics:
            test_df = pd.DataFrame(test_metrics)
            test_df['temp'] = test_df['task'] * num_epochs + test_df['epoch']
            train_df['temp'] = train_df['task'] * num_epochs + train_df['epoch']  
            df = train_df.merge(test_df, on='temp', suffixes=[None, '_dup'])
            df = df.drop('temp', 1).drop('epoch_dup', 1).drop('task_dup', 1)   
            # for continual learning, will see multiple repeats of the same epoch                              
        else:
            df = train_df
             
        df.to_csv(self.log_dir + '/' + file, sep='\t')





