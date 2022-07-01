from collections import defaultdict
import os
import pdb
from turtle import update

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import numpy as np
from tqdm import tqdm
import time
from sklearn.model_selection import ParameterGrid

from arguments import get_args, update_args, init_args, Namespace
from augmentations import get_aug
from models import get_model, get_num_params, get_head, get_features
from tools import AverageMeter, knn_monitor, probe_monitor, logistic_monitor, Logger, file_exist_check
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

def probe_evaluate(args, t, dataset, model, device, memory_loader, all_probe_results, all_probe_train_results, end_task=True):
  probe_train_results = []
  probe_results = []
  for i in range(len(dataset.test_loaders)):
    train_acc, acc, best_c = logistic_monitor(model.net.backbone, dataset, dataset.memory_loaders[i], dataset.test_loaders[i], device, args.cl_default, task_id=t, k=min(args.train.knn_k, len(memory_loader.dataset)), debug=args.debug and args.debug_lpft) 
    probe_results.append(acc)
    probe_train_results.append(train_acc)
    if not args.debug_lpft and "tune" in os.environ["logging"]: 
      tune.report(**{f"probe_acc_task_{i+1}": acc})
      tune.report(**{f"probe_train_acc_task_{i+1}": train_acc})
      tune.report(**{f"probe_best_c_{i+1}": best_c})
    if args.debug_lpft:
      print({f"probe_acc_task_{i+1}": acc})
      print({f"probe_train_acc_task_{i+1}": train_acc})
      print({f"probe_best_c_{i+1}": best_c})
    if not args.debug_lpft and "wandb" in os.environ["logging"]: 
      wandb.log({f"probe_acc_task_{i+1}": acc})
      wandb.log({f"probe_train_acc_task_{i+1}": train_acc})
      wandb.log({f"probe_best_c_{i+1}": best_c})  

  if end_task:    
    all_probe_results.append(probe_results)
    all_probe_train_results.append(probe_train_results)
    if args.train.naive:
      mean_acc = np.mean([all_probe_results[i][i] for i in range(len(dataset.test_loaders))])
      mean_train_acc = np.mean([all_probe_train_results[i][i] for i in range(len(dataset.test_loaders))])
    else:
      mean_acc = np.mean(probe_results)
      mean_train_acc = np.mean(probe_train_results)
    if not args.debug_lpft and "tune" in os.environ["logging"]: 
      tune.report(**{f"probe_train_mean_acc": mean_train_acc})
      tune.report(**{f"probe_mean_acc": mean_acc})
    if args.debug_lpft:
      print({f"probe_mean_acc": mean_acc})
      print({f"probe_train_mean_acc": mean_train_acc})
    if not args.debug_lpft and "wandb" in os.environ["logging"]: 
      wandb.log({f"probe_mean_acc": mean_acc})
      wandb.log({f"probe_train_mean_acc": mean_train_acc})


def evaluate(model: ContinualModel, dataset: ContinualDataset, device, classifier=None, fc=None, debug=False) -> Tuple[list, list]:
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
      for (inputs, labels, *meta_args) in tqdm(test_loader):
          inputs, labels = inputs.to(device), labels.to(device)            
          if fc:             
            outputs = get_features(model, inputs)
          else:
            outputs = model(inputs)
          
          if classifier is not None:
            outputs = classifier(outputs)
          elif fc is not None:
            outputs = fc[k](outputs)

          _, pred = torch.max(outputs.data, 1)
          correct += torch.sum(pred == labels).item()
          total += labels.shape[0]

          if dataset.SETTING == 'class-il' or dataset.SETTING == 'domain-il':              
            mask_classes(outputs, dataset, k if dataset.SETTING == 'class-il' else 0)
            _, pred = torch.max(outputs.data, 1)
            correct_mask_classes += torch.sum(pred == labels).item()

          if debug and total: break
      
      accs.append(correct / total * 100)
      accs_mask_classes.append(correct_mask_classes / total * 100)

    model.train(status)
    return accs, accs_mask_classes


def save_model(model, args, t, epoch, dataset):
  if args.debug_lpft:
    return
  model_path = os.path.join(args.ckpt_dir, f"{args.model.cl_model}_{t}.pth")
  if args.save_as_orig:
    model_path = os.path.join(args.ckpt_dir, f"{args.model.cl_model}_{t}_orig.pth")
  elif args.last:
    model_path = os.path.join(args.ckpt_dir, f"{args.model.cl_model}_{t}_last.pth")  
    

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

  extract_name = lambda x: x[0].split('.')[0] if args.cl_default else '.'.join(x[0].split('.')[1:3])

  num_frozen = 0
  frozen = []
  norm_avg = defaultdict(int)
  norm_avg_counts = defaultdict(int)
  all_params = list((model.net.backbone if args.cl_default else model).named_parameters())
  if not args.train.freeze_include_head:
    head_name = "fc" if args.cl_default else "predictor"
    all_params = list(filter(lambda param: param[0].split('.')[0 if args.cl_default else 1] != head_name, all_params))

  for (norm, x) in model_param_filter(all_params, float("inf")): 
    norm_avg[extract_name(x)] += norm
    norm_avg_counts[extract_name(x)+"_count"] += 1  

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

  if args.cl_default:
    # at the start, when all grad is 0, freeze in this biased order
    tiebreak = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"]
  else:
    tiebreak = ['backbone.conv1', 'backbone.bn1', 'backbone.layer1', 'backbone.layer2', 'backbone.layer3', 'backbone.layer4', 'projector.layer1', 'projector.layer2', 'projector.layer3', 'predictor.layer1', 'predictor.layer2']

  if args.train.grad_by_layer:
    num_layers = int(args.train.grad_thresh)
    assert num_layers
    layer_names = sorted(norm_avg, key=lambda x: (norm_avg[x], tiebreak.index(x)))[:num_layers]
    params_to_freeze = [(0., x) for x in all_params if extract_name(x) in layer_names]
  else:
    params_to_freeze = model_param_filter(all_params, args.train.grad_thresh)
    params_to_freeze = sorted(params_to_freeze, key=lambda x: tiebreak.index(extract_name(x[1])))

  for (norm, x) in params_to_freeze:        
    x[1].requires_grad_(False)
    frozen.append(x[0])
    num_frozen += 1

  print(f"froze {num_frozen} of {len(all_params)} parameters")
  print(frozen)

  if args.train.reset_lp_lr:
    for pg in model.opt.param_groups:
      pg['lr'] = args.train.lp_lr*args.train.batch_size/256
  if not args.cl_default and args.train.proj_is_head:
    model.net.projector.requires_grad_(False)      

def unfreeze_weights(model, args):
  model.net.backbone.requires_grad_(True)          
  for pg in model.opt.param_groups:
    pg['lr'] = args.train.ft_lr*args.train.batch_size/256
  if not args.cl_default:
    model.net.projector.requires_grad_(True)
    model.net.predictor.requires_grad_(True)


def trainable(config):
  # WANDB    
  
  args = config["default_args"]
  device = args["device"]
  
  for (k, v) in config['train'].items(): 
    args['train'] = update_args(args['train'], k, v)



    if k in args["aug_kwargs"]:
      args["aug_kwargs"][k] = v
  
  args['train'] = update_args(args['train'], 'stop_at_epoch', args['train'].num_epochs)

  assert not args['train'].all_tasks_num_epochs or not args['train'].probe_monitor  
    
  args = init_args(args)

  if args.train.disable_logging:
    os.environ['logging'] = ""
  else:
    # os.environ['logging'] = "wandb,tune"
    os.environ['logging'] = "wandb"
  if not args.debug_lpft and "wandb" in os.environ["logging"]:
    wandb.init(project=args.project_name, name=args.run_name, 
                group=args.group_name, config=Namespace.namespace_to_dict(deepcopy(args)))

  # makes fraction of lp epochs compatible with lr scheduler
  # args.train.warmup_epochs = int(args.train.warmup_lp_epoch_f * args.train.num_epochs)

  dataset = get_dataset(args)
  dataset_copy = get_dataset(args)
  train_loader, memory_loader, test_loader = dataset_copy.get_data_loaders(args, divide_tasks=args.train.all_tasks_num_epochs > 0)

  # define model
  model = get_model(args, device, len(train_loader), dataset, dataset.get_transform(args))

  backbone_n_params = get_num_params(model.net.backbone)
  if not args.debug_lpft and "tune" in os.environ["logging"]: tune.report(backbone_n_params=backbone_n_params)
  if not args.debug_lpft and "wandb" in os.environ["logging"]: wandb.log({'backbone_n_params': backbone_n_params})

  logger = Logger(matplotlib=args.logger.matplotlib, log_dir=args.log_dir)
  accuracy = 0 
  

  if args.last:
    model_path = os.path.join(args.ckpt_dir, f"{args.model.cl_model}_{3}_orig.pth")
    save_dict = torch.load(model_path, map_location='cpu')
    msg = model.net.module.backbone.load_state_dict({k[16:]:v for k, v in save_dict['state_dict'].items() if 'backbone.' in k and 'fc' not in k}, strict=True) 
    model.net.opt.load_state_dict(save_dict['opt_state_dict'])   

  old_fcs = []
  all_task_results = []

  all_probe_results = []
  all_probe_train_results = []

  for t in range(dataset.N_TASKS):
    best_current_task = float("-inf")
    train_loader, memory_loader, test_loader = dataset.get_data_loaders(args, divide_tasks=args.train.all_tasks_num_epochs == 0)
    if args.last and t < 4: 
      print("continuing cause only train last task...")
      continue

    if args.train.all_tasks_num_epochs and t == dataset.N_TASKS - 1:
      global_progress = tqdm(range(0, args.train.all_tasks_num_epochs), desc=f'Training all tasks')
    else:
      global_progress = tqdm(range(0, args.train.stop_at_epoch), desc=f'Training')

    print("at start of task, cuda allocated", print("knn cuda allocated", torch.cuda.memory_allocated()))

    epoch = 0
    for epoch in global_progress:   
      if args.lpft and (not args.train.ft_first or t):    
        freeze_weights(model, args, only_log=epoch)          
        if epoch == args.train.num_lp_epochs:
          unfreeze_weights(model, args)
      if not args.train.train_first or not t:
        model.train()
      else:
        model.eval()      
      
      local_progress=tqdm(train_loader, desc=f'Epoch {epoch}/{args.train.num_epochs}', disable=args.hide_progress)

      print(f"before training batch in epoch {epoch}, cuda allocated", torch.cuda.memory_allocated())

      for idx, ((images1, images2, notaug_images), labels, *meta_args) in enumerate(local_progress):
        data_dict = model.observe(images1, labels, images2, notaug_images)

        if not args.debug_lpft and "tune" in os.environ["logging"]: tune.report(loss=data_dict['loss'].item())
        if not args.debug_lpft and "wandb" in os.environ["logging"]: wandb.log({'loss': data_dict['loss'].item()})
        if idx == 0:
          print("after first batch, cuda allocated", torch.cuda.memory_allocated())        
        elif idx == 1:
          print("after second batch, cuda allocated", torch.cuda.memory_allocated())        
          if args.debug_lpft: break
      


      global_progress.set_postfix(data_dict)
      del data_dict

      print(f"after looping all batches in epoch {epoch}, cuda allocated", torch.cuda.memory_allocated())           

      if args.train.knn_monitor and epoch % args.train.knn_interval == 0: 
        results, results_mask_classes = [], []
        for i in range(len(dataset.test_loaders)):
          acc, acc_mask = knn_monitor(model.net.backbone, dataset, dataset.memory_loaders[i], dataset.test_loaders[i], device, args.cl_default, task_id=t, k=min(args.train.knn_k, len(memory_loader.dataset)), debug=args.debug and args.debug_lpft)             

          results.append(acc)
          if not args.debug_lpft and "tune" in os.environ["logging"]: tune.report(**{f"knn_acc_task_{i+1}": acc})
          if args.debug_lpft:
            print({f"knn_acc_task_{i+1}": acc})
          if not args.debug_lpft and "wandb" in os.environ["logging"]: wandb.log({f"knn_acc_task_{i+1}": acc})
        if not epoch:
          all_task_results.append(results)
        else:
          all_task_results[-1] = results
        if args.train.naive:
          mean_acc = np.mean([all_task_results[i][i] for i in range(len(dataset.test_loaders))])
        else:
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

      if args.train.probe_monitor and epoch % args.train.probe_interval == 0:
        probe_evaluate(args, t, dataset, model, device, memory_loader, all_probe_results, all_probe_train_results, end_task=False)

      ## BELOW for task-il evaluation, not including for domain-il

      if args.cl_default:  
        old_fcs.append(deepcopy(get_head(model.net.backbone)))      
        accs = evaluate(model.net.backbone, dataset, device, debug=args.debug and args.debug_lpft)
        # results_mask_classes.append(accs[1])
        mean_acc = accs[0]
        if args.train.all_tasks_num_epochs > 0:
          # use the last fc each test task         
          old_fcs = [old_fcs[-1] for _ in range(dataset.N_TASKS)]
        breakpoint()
        task_accs = evaluate(model.net.backbone, dataset, device, fc=old_fcs, debug=args.debug and args.debug_lpft)
        mean_acc_task_il = np.mean(task_accs,axis=1)

        if not args.debug_lpft and "tune" in os.environ["logging"]: 
          tune.report(class_il_mean_acc=mean_acc[0])
          tune.report(task_il_mean_acc=mean_acc_task_il[1])
          for i in range(len(task_accs)):
            tune.report(**{f"task_acc_{i}": task_accs[1][i]})
        if not args.debug_lpft and "wandb" in os.environ["logging"]: 
          wandb.log({'class_il_mean_acc': mean_acc[0]})
          wandb.log({'task_il_mean_acc': mean_acc_task_il[1]})
          for i in range(len(task_accs)):
            wandb.log({f'task_il_acc_{i}': task_accs[1][i]})
        if args.debug_lpft:
          print({'class_il_mean_acc': mean_acc[0]})
          print({'task_il_mean_acc': mean_acc_task_il[1]})
        # print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)


    if not args.train.save_best:
      save_model(model,args,t,epoch,dataset)
    

    if not args.train.all_tasks_num_epochs or t == dataset.N_TASKS - 1:
      # always do a probe evaluate at end of task
      probe_evaluate(args, t, dataset, model, device, memory_loader, all_probe_results, all_probe_train_results)
      





  tune.report(done=1)

  if args.eval is not False and args.cl_default is False:
      args.eval_from = model_path

  wandb.alert(
    title="Done", 
    text=f"Run with train config {config['train']} finished"
  )



  ##### RUNNNNNNN THISSSSSS!


def train(args):  
  config = {"default_args": vars(args), "train": vars(args.train)}
    ## RAY TUNE
  # tune.run(trainable, config=config, num_samples=1, resources_per_trial={"cpu": 15, "gpu": 1})
  if args.debug_lpft:
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    try:
      trainable(config=config)
    except Exception as e:
      pdb.post_mortem()
  else:
    trainable(config=config)    
    # config['train'] = {k: tune.grid_search(v) for (k, v) in config['train'].items()}
    # tune.run(trainable, config=config, num_samples=1, resources_per_trial={"cpu": 10, "gpu": 1})
    
  


if __name__ == "__main__":
    args = get_args()
    train(args=args)
    completed_log_dir = args.log_dir.replace('in-progress', 'debug' if args.debug else 'completed')
    os.rename(args.log_dir, completed_log_dir)
    print(f'Log file has been saved to {completed_log_dir}')


