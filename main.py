from collections import defaultdict
import os
import pdb
from sklearn.model_selection import ParameterGrid
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import numpy as np
from tqdm import tqdm
import time
from arguments import get_args, update_args, init_args
from augmentations import get_aug
from models import get_model
from tools import AverageMeter, knn_monitor, probe_monitor, Logger, file_exist_check
from datasets import get_dataset
from datetime import datetime
from utils.loggers import *
from utils.metrics import mask_classes
from utils.loggers import CsvLogger
from datasets.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel
from models.utils.gradient import *
from typing import Tuple
from copy import deepcopy
import os

from ray import tune
import wandb

def evaluate(model: ContinualModel, dataset: ContinualDataset, device, classifier=None, fc=None) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    assert not classifier or not fc
    if fc: assert isinstance(fc, list) and len(fc) == len(dataset.test_loaders)
    status = model.training
    model.eval()
    accs, accs_mask_classes = [], []
    for k, test_loader in enumerate(dataset.test_loaders):
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model.embed(inputs)[0] if fc else model(inputs)
            if classifier is not None:
              outputs = classifier(outputs)
            elif fc is not None:
              outputs = fc[k](outputs)

            _, pred = torch.max(outputs.data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]

            if dataset.SETTING == 'class-il':
                mask_classes(outputs, dataset, k)
                _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()
        
        accs.append(correct / total * 100)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    model.train(status)
    return accs, accs_mask_classes


def save_model(model, args, t, epoch, dataset):
  if args.debug_lpft:
    return
  model_path = os.path.join(args.ckpt_dir, f"{args.model.cl_model}_{args.name}_{t}.pth")
  if args.save_as_orig:
    model_path = os.path.join(args.ckpt_dir, f"{args.model.cl_model}_{args.name}_{t}_orig.pth")
  elif args.last:
    model_path = os.path.join(args.ckpt_dir, f"{args.model.cl_model}_{args.name}_{t}_last.pth")  
    

  torch.save({
    'epoch': epoch+1,
    'state_dict':model.net.state_dict(),
    'opt_state_dict':model.opt.state_dict()
  }, model_path)

  print(f"Task Model saved to {model_path}")

  with open(os.path.join(args.log_dir, f"checkpoint_path.txt"), 'w+') as f:
    f.write(f'{model_path}')
  
  if hasattr(model, 'end_task'):
    model.end_task(dataset)

def freeze_weights(model, args, only_log=False):
  num_frozen = 0
  frozen = []
  norm_avg = defaultdict(int)
  norm_avg_counts = defaultdict(int)
  all_params = list((model.net.module.backbone if args.cl_default else model).named_parameters())
  for (norm, x) in model_param_filter(model.net.module.backbone if args.cl_default else model, float("inf")): 
    norm_avg[x[0].split('.')[0]] += norm
    norm_avg_counts[x[0].split('.')[0]+"_count"] += 1  

  for k in norm_avg:
    norm_avg[k] /= norm_avg_counts[k+"_count"]

  if len(norm_avg):
    for k in norm_avg:
      if not args.debug_lpft and "tune" in os.environ["logging"]: 
        tune.report(**{f"grad/{k}_grad_abs_mean": float(norm_avg[k])})
      if not args.debug_lpft and "wandb" in os.environ["logging"]: 
        wandb.log({f"grad/{k}_grad_abs_mean": float(norm_avg[k])})
      if args.debug_lpft:
        print({f"grad/{k}_grad_abs_mean": float(norm_avg[k])})

  if only_log: return

  tiebreak = ["layer4", "layer3", "layer2", "layer1", "bn1", "conv1", "fc"]

  if args.train.grad_by_layer:
    num_layers = int(args.train.grad_thresh)
    assert num_layers
    layer_names = sorted(norm_avg, key=lambda x: (norm_avg[x], tiebreak.index(x)))[:num_layers]
    params_to_freeze = [(0., x) for x in (model.net.module.backbone if args.cl_default else model).named_parameters() if x[0].split('.')[0] in layer_names]
  else:
    params_to_freeze = model_param_filter(model.net.module.backbone if args.cl_default else model, args.train.grad_thresh)
    # move fc ones last
    params_to_freeze = sorted(params_to_freeze, key=lambda x: tiebreak.index(x[0].split('.')))

  for (norm, x) in params_to_freeze:        
    if frozen or "fc" not in x[0]:
      # don't freeze fc if no param has been frozen
      x[1].requires_grad_(False)
      frozen.append(x[0])
      num_frozen += 1

  print(f"froze {num_frozen} of {len(all_params)} parameters")
  print(frozen)

  if args.train.reset_lp_lr:
    for pg in model.opt.param_groups:
      pg['lr'] = args.train.lp_lr*args.train.batch_size/256
  if not args.cl_default and args.train.proj_is_head:
    model.net.module.projector.requires_grad_(False)      

def unfreeze_weights(model, args):
  model.net.module.backbone.requires_grad_(True)          
  for pg in model.opt.param_groups:
    pg['lr'] = args.train.ft_lr*args.train.batch_size/256
  if not args.cl_default:
    model.net.module.projector.requires_grad_(True)
    model.net.module.predictor.requires_grad_(True)


def trainable(config):
  # WANDB    
  
  args = config["default_args"]
  device = args["device"]
  for (k, v) in config['train'].items(): 
    args['train'] = update_args(args['train'], k, v)

  os.environ['logging'] = "wandb,tune"
  if not args['debug_lpft']:
    wandb.init(project="lpft", config=vars(args['train']))
    
  args = init_args(args)

  # makes fraction of lp epochs compatible with lr scheduler
  # args.train.warmup_epochs = int(args.train.warmup_lp_epoch_f * args.train.num_epochs)

  dataset = get_dataset(args)
  dataset_copy = get_dataset(args)
  train_loader, memory_loader, test_loader = dataset_copy.get_data_loaders(args)

  # define model
  model = get_model(args, device, len(train_loader), dataset.get_transform(args))

  logger = Logger(matplotlib=args.logger.matplotlib, log_dir=args.log_dir)
  accuracy = 0 
  

  if args.last:
    model_path = os.path.join(args.ckpt_dir, f"{args.model.cl_model}_{args.name}_{3}_orig.pth")
    save_dict = torch.load(model_path, map_location='cpu')
    msg = model.net.module.backbone.load_state_dict({k[16:]:v for k, v in save_dict['state_dict'].items() if 'backbone.' in k and 'fc' not in k}, strict=True) 
    model.net.opt.load_state_dict(save_dict['opt_state_dict'])   

  old_fcs = []

  for t in range(dataset.N_TASKS):
    best_current_task = float("-inf")
    train_loader, memory_loader, test_loader = dataset.get_data_loaders(args)
    if args.last and t < 4: 
      print("continuing cause only train last task...")
      continue

    global_progress = tqdm(range(0, args.train.stop_at_epoch), desc=f'Training')
    for epoch in global_progress:   
      if args.lpft and (not args.train.ft_first or t):    
        freeze_weights(model, args, only_log=epoch)          
        if epoch == args.train.num_lp_epochs:
          unfreeze_weights(model, args)
      if not args.train.train_first or not t:
        model.train()
      else:
        model.eval()
      results, results_mask_classes = [], []
      
      local_progress=tqdm(train_loader, desc=f'Epoch {epoch}/{args.train.num_epochs}', disable=args.hide_progress)

      for idx, ((images1, images2, notaug_images), labels) in enumerate(local_progress):
          data_dict = model.observe(images1, labels, images2, notaug_images)
          if not args.debug_lpft and "tune" in os.environ["logging"]: tune.report(loss=data_dict['loss'].item())
          if not args.debug_lpft and "wandb" in os.environ["logging"]: wandb.log({'loss': data_dict['loss'].item()})
          if args.debug_lpft:
            break

      global_progress.set_postfix(data_dict)

      if args.train.knn_monitor and epoch % args.train.knn_interval == 0: 
          for i in range(len(dataset.test_loaders)):
            
            acc, acc_mask = knn_monitor(model.net.module.backbone, dataset, dataset.memory_loaders[i], dataset.test_loaders[i], device, args.cl_default, task_id=t, k=min(args.train.knn_k, len(memory_loader.dataset)))             

            results.append(acc)
            if not args.debug_lpft and "tune" in os.environ["logging"]: tune.report(**{f"knn_acc_task_{i+1}": acc})
            if args.debug_lpft:
              print({f"knn_acc_task_{i+1}": acc})
            if not args.debug_lpft and "wandb" in os.environ["logging"]: wandb.log({f"knn_acc_task_{i+1}": acc})
          mean_acc = np.mean(results)
          if not args.debug_lpft and "tune" in os.environ["logging"]: tune.report(**{f"knn_mean_acc": mean_acc})
          if args.debug_lpft:
            print({f"knn_mean_acc": mean_acc})
          if not args.debug_lpft and "wandb" in os.environ["logging"]: wandb.log({f"knn_mean_acc": mean_acc})


        
      epoch_dict = {"epoch":epoch, "accuracy": mean_acc}
      print("mean_accuracy:", mean_acc)
      global_progress.set_postfix(epoch_dict)
      logger.update_scalers(epoch_dict)

      if args.train.save_best and results[-1] > best_current_task:
        print(f"{results[-1]} beats {best_current_task}, saving model...")
        best_current_task = results[-1]
        save_model(model,args,t,epoch,dataset)

    if not args.train.save_best:
      save_model(model,args,t,epoch,dataset)
    
    if args.cl_default:
      old_fcs.append(deepcopy(model.net.module.backbone.fc))
      accs = evaluate(model.net.module.backbone, dataset, device)
      results.append(accs[0])
      results_mask_classes.append(accs[1])
      mean_acc = np.mean(accs, axis=1)

      task_accs = evaluate(model.net.module.backbone, dataset, device, fc=old_fcs)
      mean_acc_task_il = np.mean(task_accs,axis=1)

      if not args.debug_lpft and "tune" in os.environ["logging"]: 
        tune.report(class_il_mean_acc=mean_acc[0])
        tune.report(task_il_mean_acc=mean_acc_task_il[1])
      if not args.debug_lpft and "wandb" in os.environ["logging"]: 
        wandb.log({'class_il_mean_acc': mean_acc[0]})
        wandb.log({'task_il_mean_acc': mean_acc_task_il[1]})
      if args.debug_lpft:
        print({'class_il_mean_acc': mean_acc[0]})
        print({'task_il_mean_acc': mean_acc_task_il[1]})
      # print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)

      probe_results = []
      for i in range(len(dataset.test_loaders)):
        acc, acc_mask = probe_monitor(model.net.module.backbone, dataset, dataset.memory_loaders[i], dataset.test_loaders[i], device, args.cl_default, task_id=t, k=min(args.train.knn_k, len(memory_loader.dataset))) 
        probe_results.append(acc)
        if not args.debug_lpft and "tune" in os.environ["logging"]: tune.report(**{f"probe_acc_task_{i+1}": acc})
        if args.debug_lpft:
          print({f"probe_acc_task_{i+1}": acc})
        if not args.debug_lpft and "wandb" in os.environ["logging"]: wandb.log({f"probe_acc_task_{i+1}": acc})
      mean_acc = np.mean(probe_results)
      if not args.debug_lpft and "tune" in os.environ["logging"]: tune.report(**{f"probe_mean_acc": mean_acc})
      if args.debug_lpft:
        print({f"probe_mean_acc": mean_acc})
      if not args.debug_lpft and "wandb" in os.environ["logging"]: wandb.log({f"probe_mean_acc": mean_acc})
        





  tune.report(done=1)

  if args.eval is not False and args.cl_default is False:
      args.eval_from = model_path

  wandb.alert(
    title="Done", 
    text=f"Run with train config {config['train']} finished"
  )


def train(args):  
  config = {"default_args": vars(args), "train": {
    # "save_best": [True],
    "cl_default": [True],
    # "warmup_epochs": [10],
    # "warmup_lr": [0],
    # "lp_lr": [0.03],
    # "ft_lr": [0.03],
    "grad_thresh": [0.],
    # "grad_by_layer": [True],
    "num_lp_epochs": [0],
    # "num_epochs": [2],
    # "stop_at_epoch": [2],
    # "proj_is_head": [False],
  }}
    ## RAY TUNE
  # tune.run(trainable, config=config, num_samples=1, resources_per_trial={"cpu": 15, "gpu": 1})
  if args.debug_lpft:
    config['train'] = ParameterGrid(config['train'])[0]
    try:
      trainable(config=config)
    except Exception as e:
      pdb.post_mortem()
  else:
    config['train'] = {k: tune.grid_search(v) for (k, v) in config['train'].items()}
    tune.run(trainable, config=config, num_samples=1, resources_per_trial={"cpu": 15, "gpu": 1})
    
  


if __name__ == "__main__":
    args = get_args()
    train(args=args)
    completed_log_dir = args.log_dir.replace('in-progress', 'debug' if args.debug else 'completed')
    os.rename(args.log_dir, completed_log_dir)
    print(f'Log file has been saved to {completed_log_dir}')


