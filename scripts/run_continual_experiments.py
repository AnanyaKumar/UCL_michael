import argparse
import ast
from collections import namedtuple
from itertools import product
from sklearn.model_selection import ParameterGrid
from copy import deepcopy
import re
import shlex
import subprocess

PROJECT_NAME = "continual_learning"

Dataset = namedtuple(
    'Dataset',
    ['name', 'val_metric', 'output_metrics', 'config_rel_path', 'slurm_data_cmd']
)

cifar10_scl = Dataset(
    name='cifar10_scl',
    val_metric='knn_mean_acc',
    output_metrics=['knn_acc_task_0', 'knn_acc_task_1', 'knn_acc_task_2', 'knn_acc_task_3', 'knn_acc_task_4', 'knn_mean_acc'],
    config_rel_path='finetune_c10_scl.yaml',
    slurm_data_cmd=None
)

cifar10 = Dataset(
    name='cifar10',
    val_metric='knn_mean_acc',
    output_metrics=['knn_acc_task_0', 'knn_acc_task_1', 'knn_acc_task_2', 'knn_acc_task_3', 'knn_acc_task_4', 'knn_mean_acc'],
    config_rel_path='simsiam_c10.yaml',
    slurm_data_cmd=None
)

fmow = Dataset(
    name='fmow',
    val_metric='task_il_mean_acc',
    output_metrics=['task_il_mean_acc'],
    config_rel_path='finetune_fmow_scl.yaml',
    slurm_data_cmd='scripts/copy_dataset.sh'
)

names_to_datasets = {
    'cifar10_scl': cifar10_scl, 
    'cifar10': cifar10,
    "fmow": fmow
}

def get_dataset(name):
    return names_to_datasets[name]


def process(d):
    """https://stackoverflow.com/questions/36198540/split-python-dictionary-to-result-in-all-combinations-of-values"""
    to_product = []  # [[('a', 1), ('a', 2)], [('b', 3),], ...]
    for k, v in d.items():
        if isinstance(v, list):
            to_product.append([(k, i) for i in v])
        elif isinstance(v, dict):
            to_product.append([(k, i) for i in process(v)])
        else:
            to_product.append([(k, v)])
    return [dict(l) for l in product(*to_product)]


def transform_unparsed(unparsed):
    unparsed_dic = {}
    for unparsed_option in unparsed:
        option_name, val = unparsed_option.split('=')
        # get rid of --
        option_name = option_name[2:].strip()
        # handle nesting
        option_name_list = option_name.split('.')
        val = val.split(',')
        vals = []

        # interpret the string as int, float, string, bool, etc
        for v in val:
            try:
                v = ast.literal_eval(v.strip())
            except Exception:
                # keep as string
                v = v.strip()
            vals.append(v)
        
        curr_dict = unparsed_dic 

        for k in option_name_list[:-1]:
            try: 
                curr_dict = curr_dict[k]
            except KeyError:
                curr_dict[k] = {}
                curr_dict = curr_dict[k]
        
        curr_dict[option_name_list[-1]] = vals
    
    results = []
    all_res = process(unparsed_dic)

    for res in all_res:
        for (k, v) in res['train'].items():
            res['train.'+k] = v
        res.pop('train')
        results.append(res)
    
    return all_res


############################################
## Functions to get directory/job names.
############################################

def hyperparams_to_str(hyperparams, item_sep='_', key_value_sep='-', ignore_name_hypers={}):
    """Convert hyperparameters into string."""
    sorted_hyperparams = sorted(hyperparams.items())
    return item_sep.join([str(k) + key_value_sep + str(v) for k, v in sorted_hyperparams
                          if k not in ignore_name_hypers])

def get_config_path(args, config_rel_path):
    return args.config_dir + '/' + config_rel_path

def group_run_to_log_path(group_name, run_name, args):
    return args.log_dir + '/' + group_name + '/' + run_name
 
def format_key_value(k, v):
    if type(v) == list:
        if type(v[0]) == list:
            raise ValueError('We only support 1D lists.')
        return f'--{k} ' + ' '.join([str(e) for e in v])
    # I wanted to do this, but this messes up with update_config in utils.py, and hard to fix that.
    # if type(v) == bool:
    #     if v:
    #         return f'--{k}'
    #     return ''
    return f'--{k}=' + str(v)

def get_python_cmd(code_path, python_path='python', kwargs=None, args=None):
    if kwargs is not None:
        # Make sure to keep the space at the end.
        opts = ''.join([f"{format_key_value(k, v)} " for k, v in kwargs.items()])
        # opts += ''.join([f"--{k} " for k, v in kwargs.items() if isinstance(v, bool) and v and '.' not in k])
    else:
        opts = ''
    python_cmd = python_path + ' ' + code_path + ' '
    python_cmd += opts

    return python_cmd


def get_baseline_experiment_cmd(config_path, run_name, group_name, project_name, kwargs, args):
    # If run_saved, then we ignore root_prefix since we get this from the config.
    kwargs = deepcopy(kwargs)
    # Sometimes we might want to run from a saved json config file, in a custom location.
    # Saved files have full dataset paths, e.g. /scr/biggest/..., so no need to add root_prefix.
    kwargs['config'] = config_path
   
    kwargs['log_dir'] = args.log_dir
    kwargs['tmp_par_ckp_dir'] = args.tmp_dir + '/' + group_name + '_' + run_name

    kwargs['project_name'] = project_name
    kwargs['group_name'] = group_name
    kwargs['run_name'] = run_name
    code_path = args.code_dir + '/' + 'main.py'
    return (get_python_cmd(code_path=code_path, python_path=args.python_path, kwargs=kwargs,
                           args=args),
            kwargs['log_dir'])

def run_sbatch(cmd, job_name, args):
    output_path = args.output_dir + '/' + job_name
    sbatch_script_path = args.scripts_dir + '/' + args.sbatch_script_name 
    slurm_cmd = f'sbatch --partition={args.partition} --job-name={job_name} --output={output_path} '
    slurm_cmd += f' {sbatch_script_path} '
    slurm_cmd += f'"{cmd}"'
    print(slurm_cmd + '\n')
    output = subprocess.check_output(shlex.split(slurm_cmd)).decode('utf8')
    job_names = list(re.findall(r'\d+', output))
    assert(len(job_names) == 1)
    return job_names[0]


def run_job(cmd, job_name, args):
    return run_sbatch(cmd, job_name, args)

def config_run(args, kwargs, config_path, run_name, group_name, project_name,
               dataset_copy_cmd=None):
    cmd, log_dir = get_baseline_experiment_cmd(
        config_path=config_path, run_name=run_name, group_name=group_name,
        project_name=project_name, kwargs=kwargs, args=args)
    if dataset_copy_cmd is not None:
        cmd = dataset_copy_cmd + ' && ' + cmd
    job_name = group_name + '_' + run_name
    return run_job(cmd, job_name, args)


def run_adapt_sweep(adapt_name, dataset, hyperparams, args, run_name_suffix=''):
    run_name = hyperparams_to_str(hyperparams)
    run_name += run_name_suffix
    group_name = adapt_name.split("_")[0]
    project_name = PROJECT_NAME
    kwargs = deepcopy(hyperparams)
    config_path = get_config_path(args, dataset.config_rel_path)
    dataset_copy_cmd = None
    if dataset.slurm_data_cmd is not None:
        dataset_copy_cmd = dataset.slurm_data_cmd.format(scripts_dir=args.scripts_dir)

    return config_run(args, kwargs=kwargs, config_path=config_path,
        run_name=run_name, group_name=group_name, project_name=project_name,
        dataset_copy_cmd=dataset_copy_cmd)
    

def replicated_sweep(adapt_name, dataset, hyperparams_list, num_replications,
                     args):
    # Run multiple replications for each sweep run.
    sweep_ids = []
    for i in range(num_replications):
        for hyperparams in hyperparams_list: 
            kwargs = deepcopy(hyperparams)
            kwargs['seed'] = args.seed + i
            job_id = run_adapt_sweep(adapt_name, dataset,
                hyperparams=kwargs, args=args, run_name_suffix='_run'+str(i))
            # Job id of -1 means we didn't run the job because it's already run.
            if job_id != -1:
                sweep_ids.append(job_id)
    return sweep_ids 


def lpft_experiments(args, unparsed, num_replications=1):
    adapt_name = 'lpft'
    dataset = get_dataset(args.dataset)
    adapt_name += f"_{dataset.name}"
    hyperparameters_list = transform_unparsed(unparsed)

    if args.only_one_run:
        hyperparameters_list = [hyperparameters_list[0]]
    
    all_ids = replicated_sweep(
        adapt_name=adapt_name, dataset=dataset, hyperparams_list=hyperparameters_list,
        num_replications=num_replications, args=args)
    
    print(all_ids)


def main(args, unparsed):
    experiment_to_fns = {
        'lpft': lpft_experiments,
    }
    if args.experiment in experiment_to_fns:
        experiment_to_fns[args.experiment](args, unparsed)
    else:
        raise ValueError(f'Experiment {args.experiment} does not exist.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run continual experiments.')
    parser.add_argument('--experiment', type=str, required=True,
                        help='Experiment to run.')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset to train and test on.')
    parser.add_argument('--seed', type=int, required=False, default=0,
                        help='Base seed, we typically add to this seed for replication runs.')
    # Note that store_true creates a default value of False.
    parser.add_argument('--partition', type=str, required=False, default='jag-standard',
                        help='(Slurm only) What priority to use.')
    # Locations of folders and files.
    parser.add_argument('--scripts_dir', type=str, required=False, default='scripts/',
                        help='Path to dir where scripts are stored.')
    parser.add_argument('--output_dir', type=str, required=False, default='slurm_outputs/',
                        help='(Slurm only) Path to dir to store stdout for experiment.')
    parser.add_argument('--log_dir', type=str, required=False, default='logs/',
                        help='Path to dir where we save logs and run checkpoints.')
    parser.add_argument('--config_dir', type=str, required=False, default='configs/',
                        help='Directory where config files are stored.')
    parser.add_argument('--code_dir', type=str, required=False, default='.',
                        help='Path to directory where main.py file is located.')
    parser.add_argument('--tmp_dir', type=str, required=False, default='/scr/biggest/ue_michael/',
                        help='(Slurm only) Directory where tmp files are stored.')
    parser.add_argument('--python_path', type=str, required=False, default='python',
                        help='Path or alias to Python interpreter')
    parser.add_argument('--sbatch_script_name', type=str, required=False, default='run_sbatch.sh',
                        help='(Slurm only) sbatch script')        
    parser.add_argument('--only_one_run', action='store_true',
                        help=('Only run one hyperparameter setting, e.g. for debugging'
                              '(also do not run replications).'), required=False)
    args, unparsed = parser.parse_known_args()
    main(args, unparsed)