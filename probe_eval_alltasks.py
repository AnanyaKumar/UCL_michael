import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
from tqdm import tqdm
from arguments import get_args, init_args
from augmentations import get_aug
from models import get_model, get_backbone
from tools import AverageMeter, knn_monitor, probe_evaluate, Logger, logistic_monitor
from datasets import get_dataset
from models.optimizers import get_optimizer, LR_Scheduler
from utils.loggers import *
from collections import defaultdict


def knn_evaluate_single(model, dataset, test_loader, memory_loader, device, k, last=False) -> Tuple[list, list, list, list]:
    accs, accs_mask_classes = [], []
    knn_accs, knn_accs_mask_classes = [], []
    correct = correct_mask_classes = total = 0
    knn_acc, knn_acc_mask = knn_monitor(model.net.backbone, dataset, memory_loader, test_loader, device, args.cl_default, task_id=k, k=min(args.train.knn_k, len(dataset.memory_loaders[k].dataset)), debug=False) 
    
    return knn_acc


def probe_evaluate_single(model, dataset, test_loader, memory_loader, all_probe_results, all_probe_train_results, device, k, last=False) -> Tuple[list, list, list, list]:
    probe_evaluate(args, t, dataset, model, device, memory_loader, all_probe_results, all_probe_train_results, train_stats, test_stats, end_task=False)
    return knn_acc


def evaluate(model, model_path, args, device, dataset_copy, dataset, test_stats, task_str):
  test_stats['task_str'].append(task_str)
  save_dict = torch.load(model_path, map_location='cpu')
  test_stats['epoch'].append(save_dict['epoch'])
  msg = model.net.backbone.load_state_dict({k.split('backbone.')[1]:v for k, v in save_dict['state_dict'].items() if 'backbone.' in k}, strict=True)
  model = model.to(args.device)

  if args.train.knn_monitor:
    task_knn_acc = []
    for t1 in tqdm(range(0, dataset_copy.N_TASKS), desc='Inner tasks'):
      train_loader, memory_loader, test_loader = dataset_copy.train_loaders[t1], dataset_copy.memory_loaders[t1], dataset_copy.test_loaders[t1]
      t1_knn_acc = knn_evaluate_single(model, dataset_copy, test_loader, memory_loader, device, t1)
      task_knn_acc.append(t1_knn_acc)
      test_stats[f'knn_acc_task_{t1}'].append(t1_knn_acc)
    test_stats[f'knn_mean_acc'].append(np.mean(task_knn_acc))
    print(f'Task {task_str} {args.probe_train_frac}-knn probe: {task_knn_acc}')

  all_probe_acc = []
  all_probe_train_acc = []
  if args.train.probe_monitor:
    task_probe_acc = []
    task_probe_train_acc = []
    for t1 in tqdm(range(0, dataset_copy.N_TASKS), desc='Inner tasks'):
      train_loader, memory_loader, test_loader = dataset_copy.train_loaders[t1], dataset_copy.memory_loaders[t1], dataset_copy.test_loaders[t1]      
      train_acc, acc, best_c = logistic_monitor(model.net.backbone, dataset_copy, memory_loader, test_loader, device, args.cl_default, task_id=t1, k=min(args.train.knn_k, len(memory_loader.dataset)))
      task_probe_acc.append(acc)
      task_probe_train_acc.append(train_acc)
      test_stats[f'probe_acc_task_{t1}'].append(acc)
      test_stats[f'probe_acc_train_task_{t1}'].append(train_acc)
    test_stats[f'probe_mean_acc'].append(np.mean(task_probe_acc))
    test_stats[f'probe_train_mean_acc'].append(np.mean(task_probe_train_acc))
    print(f'Task {task_str} {args.probe_train_frac}-linear probe: {task_probe_acc}')


def main(device, args):
    os.environ['logging'] = ""
    config = {"default_args": vars(args), "train": vars(args.train)}
    args = config["default_args"]
    device = args["device"]    
    # Do some train.argument specific assertions here   
    args = init_args(args)
    args.aug_kwargs = vars(args.aug_kwargs)
    dataset_copy = get_dataset(args)
    train_loader, memory_loader, test_loader = dataset_copy.get_data_loaders(args)
    model = get_model(args, device, len(train_loader), dataset_copy, dataset_copy.get_transform(args))
    logger = Logger(matplotlib=args.logger.matplotlib, log_dir=args.log_dir)

    for t in range(dataset_copy.N_TASKS - 1):
        _, _, _ = dataset_copy.get_data_loaders(args)

    test_metrics = []    
    knn_acc = []
    all_probe_acc, all_probe_train_acc = [], []
    for t in tqdm(range(-1, dataset_copy.N_TASKS), desc='Evaluating'):
      test_stats = defaultdict(list)      
      if t < 0:
        task_str = f"0:{dataset_copy.N_TASKS}"        
      else:
        task_str = t 
      
      dataset = get_dataset(args)
      if os.path.exists(os.path.join(args.ckpt_dir, f"{args.model.cl_model}_{task_str}.pth")):
        model_path = os.path.join(args.ckpt_dir, f"{args.model.cl_model}_{task_str}.pth")
      elif os.path.exists(os.path.join(args.tmp_par_ckp_dir, f"checkpoints/{args.model.cl_model}_{task_str}.pth")):
        # to debug on same machine as training
        model_path = os.path.join(args.tmp_par_ckp_dir, f"checkpoints/{args.model.cl_model}_{task_str}.pth")
      else:
        breakpoint()
        raise AssertionError("checkpoint path doesn't exist")

      evaluate(model, model_path, args, device, dataset_copy, dataset, test_stats, task_str)          
      
      test_metrics.append(test_stats)
      logger.process_stats(test_stats)
      test_metrics.append(test_stats)
      logger.write_tsv(test_metrics, file=f"{args.probe_train_frac}-probe_eval.tsv")


    # mean_knn_acc = sum(knn_acc[-1][:len(knn_acc[-1])]) / len(knn_acc[-1])
    # print(f'KNN accuracy on Task {t1}: {mean_knn_acc}')

    # max_knn_acc = [max(idx) for idx in zip(*knn_acc)]
    # mean_knn_fgt = sum([x1 - x2 for (x1, x2) in zip(max_knn_acc, knn_acc[-1])]) / len(knn_acc[-1])
    # print(f'KNN Forgetting: {mean_knn_fgt}')


if __name__ == "__main__":
    args, checkpoints_dir = get_args()
    main(device=args.device, args=args)
