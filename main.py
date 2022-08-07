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
from tools import AverageMeter, knn_monitor, probe_monitor, set_probe, logistic_monitor, Logger, file_exist_check
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
import shutil
import os

from ray import tune
import wandb

def probe_evaluate(args, t, dataset, model, device, memory_loader, all_probe_results, all_probe_train_results, train_stats=None, test_stats=None, end_task=True):
  probe_train_results = []
  probe_results = []
  for i in range(len(dataset.test_loaders)):
    train_acc, acc, best_c = logistic_monitor(model.net.backbone, dataset, dataset.memory_loaders[i], dataset.test_loaders[i], device, args.cl_default, task_id=t, k=min(args.train.knn_k, len(memory_loader.dataset)))
    probe_results.append(acc)
    probe_train_results.append(train_acc)

    if test_stats: test_stats[f"probe_acc_task_{i+1}"].append(acc)
    if train_stats: train_stats[f"probe_train_acc_task_{i+1}"].append(train_acc)
    if test_stats: test_stats[f"probe_best_c_{i+1}"].append(best_c)
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


  all_probe_results.append(probe_results)
  all_probe_train_results.append(probe_train_results)
  if args.train.naive:
    mean_acc = np.mean([all_probe_results[i][i] for i in range(len(dataset.test_loaders))])
    mean_train_acc = np.mean([all_probe_train_results[i][i] for i in range(len(dataset.test_loaders))])
  else:
    mean_acc = np.mean(probe_results)
    mean_train_acc = np.mean(probe_train_results)
  if train_stats: train_stats[f"probe_train_mean_acc"].append(mean_train_acc)
  if train_stats: train_stats[f"probe_mean_acc"].append(mean_acc)
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
  if isinstance(t, list):
    task_str = f"{t[0]}:{t[-1]+1}"
  else:
    task_str = t
  if args.debug_lpft:
    return
  model_path = os.path.join(args.tmp_par_ckp_dir, f"checkpoints/{args.model.cl_model}_{task_str}.pth")
  if args.save_as_orig:
    model_path = os.path.join(args.tmp_par_ckp_dir, f"checkpoints/{args.model.cl_model}_{task_str}_orig.pth")
  elif args.last:
    model_path = os.path.join(args.tmp_par_ckp_dir, f"checkpoints/{args.model.cl_model}_{task_str}_last.pth")  
    
  if args.save_model:
    torch.save({
      'epoch': epoch+1,
      'state_dict':model.net.state_dict(),
      'opt_state_dict':model.opt.state_dict()
    }, model_path)

    print(f"Task Model saved to {model_path}")


    with open(os.path.join(args.log_dir, f"checkpoint_path.txt"), 'a+') as f:
      f.write(f'{model_path}\n')    

def layer_taxonomy(name, cl_default):
  """
    At the start, when all grad is 0, freeze in this biased order.
    Also useful to see.
  """
  if name == 'resnet18':
    extract_name = lambda x: x[0].split('.')[0] if args.cl_default else '.'.join(x[0].split('.')[1:3])
    if cl_default:    
      head_name = "fc"  
      tiebreak = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"]
    else:
      head_name = "predictor"
      tiebreak = ['backbone.conv1', 'backbone.bn1', 'backbone.layer1', 'backbone.layer2', 'backbone.layer3', 'backbone.layer4', 'projector.layer1', 'projector.layer2', 'projector.layer3', 'predictor.layer1', 'predictor.layer2']
  elif name == 'densenet121':
    extract_name = lambda x: x[0].split('.')[1]
    if cl_default:
      head_name = "classifier"
      tiebreak = ['conv0', 'norm0', 'denseblock1', 'transition1', 'denseblock2', 'transition2', 'denseblock3', 'transition3', 'denseblock4', 'norm5']
    else:
      raise NotImplementedError
  else:
    raise NotImplementedError
  
  return tiebreak, extract_name, head_name


def freeze_weights(model, args, only_log=False):
  tiebreak, extract_name, head_name = layer_taxonomy(args.model.backbone, args.cl_default)
  num_frozen = 0
  frozen = []
  norm_avg = defaultdict(int)
  norm_avg_counts = defaultdict(int)
  all_params = list((model.net.backbone if args.cl_default else model).named_parameters())
  if not args.train.freeze_include_head:
    # cause when cl_default=False all params start with prefix "backbone." (but check if true)
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
    if "resnet" not in args.model.name:
      raise NotImplementedError
    model.net.projector.requires_grad_(True)
    model.net.predictor.requires_grad_(True)


def trainable(config):
  # WANDB    
  
  args = config["default_args"]
  device = args["device"]
  
  # Do some train.argument specific assertions here
  args['train'] = update_args(args['train'], 'stop_at_epoch', args['train'].num_epochs)
  assert not args['train'].all_tasks_num_epochs or not args['train'].probe_monitor  
    
  args = init_args(args)

  args.aug_kwargs = vars(args.aug_kwargs)

  if args.train.disable_logging:
    os.environ['logging'] = ""
  else:
    # os.environ['logging'] = "wandb,tune"
    os.environ['logging'] = "wandb"
  if not args.debug_lpft and "wandb" in os.environ["logging"]:
    api = wandb.Api()
    runs = api.runs(path=f"lpft/{args.project_name}", filters={"config.group_name": args.group_name, "config.run_name": args.run_name})
    if len(runs):
      for i in range(len(runs)):
        if runs[i].state == "finished":
          if not args.rerun:
            print("Exiting, existing run already finished")
        elif runs[i].state == "crashed":
          continue
        else:
          continue

      print("Redoing run, previously no runs finished")

    user = os.environ["USER"]
    wandb.init(project=args.project_name, name=args.run_name, 
                dir=f"/nlp/scr/{user}/wandb/",
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
  
  all_task_results = []
  old_fcs = []
  all_probe_results = []
  all_probe_train_results = []

  train_metrics = []
  test_metrics = []

  for t in range(dataset.N_TASKS):
    best_current_task = float("-inf")
    best_mean_task = float("-inf")
    train_loader, memory_loader, test_loader = dataset.get_data_loaders(args, divide_tasks=args.train.all_tasks_num_epochs == 0)
    if args.last and t < dataset.N_TASKS - 1: 
      print("continuing cause only train last task...")
      continue

    if args.train.all_tasks_num_epochs and t == dataset.N_TASKS - 1:
      global_progress = tqdm(range(0, args.train.all_tasks_num_epochs), desc=f'Training all tasks')
      args.train.num_epochs = args.train.all_tasks_num_epochs
    else:
      global_progress = tqdm(range(0, args.train.stop_at_epoch), desc=f'Training')

    print(f"at start of task {t}, cuda allocated", print("knn cuda allocated", torch.cuda.memory_allocated()))    

    epoch = 0
    for epoch in global_progress: 

      ### Setup logging  
      train_stats = defaultdict(list)
      test_stats = defaultdict(list)
      train_stats['epoch'] = [epoch]
      test_stats['epoch'] = [epoch]
      train_stats['task'] = [t]
      test_stats['task'] = [t]

      if args.lpft and (not args.train.ft_first or t):    
        freeze_weights(model, args, only_log=epoch > 0)
        if epoch == args.train.num_lp_epochs:
          if args.train.sklearn_lp_probe:
            print("initializing sklearn probe")
            set_probe(model.net.backbone, dataset, memory_loader, test_loader, device, args.cl_default, t)
          
          unfreeze_weights(model, args)
      if not args.train.train_first or not t:
        model.train()
      else:
        model.eval()      
      
      local_progress=tqdm(train_loader, desc=f'Epoch {epoch}/{args.train.num_epochs}', disable=args.hide_progress)

      print(f"before training batch in epoch {epoch}, cuda allocated", torch.cuda.memory_allocated())

      for idx, ((images1, images2, notaug_images), labels, *meta_args) in enumerate(local_progress):
        data_dict = model.observe(images1, labels, images2, notaug_images)

        for (k, v) in data_dict.items():
          train_stats[k].append((epoch, float(v)))

        if not args.debug_lpft and "tune" in os.environ["logging"]: 
          tune.report(loss=data_dict['loss'].item())
        if not args.debug_lpft and "wandb" in os.environ["logging"]: 
          wandb.log({'loss': data_dict['loss'].item()})
          
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
          test_stats[f'knn_acc_task_{i+1}'].append(acc)
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

        test_stats[f"knn_mean_acc"].append(mean_acc)
        if not args.debug_lpft and "tune" in os.environ["logging"]: tune.report(**{f"knn_mean_acc": mean_acc})
        if args.debug_lpft:
          print({f"knn_mean_acc": mean_acc})
        if not args.debug_lpft and "wandb" in os.environ["logging"]: wandb.log({f"knn_mean_acc": mean_acc})
              
        epoch_dict = {"epoch":epoch, "accuracy": mean_acc}
        print("mean_accuracy:", mean_acc)
        global_progress.set_postfix(epoch_dict)
        logger.update_scalers(epoch_dict)

      if args.train.save_best:
        if results[-1] > best_current_task:
          print(f"{results[-1]} beats {best_current_task} on current task, saving model...")
          best_current_task = results[-1]
          save_model(model, args, t, epoch, dataset)
        
        if mean_acc > best_mean_task:
          print(f"{mean_acc} beats {best_mean_task} on mean acc of tasks, saving model...")
          best_mean_task = mean_acc
          save_model(model, args, list(range(t+1)), epoch, dataset)

      if args.train.probe_monitor and epoch % args.train.probe_interval == 0:
        probe_evaluate(args, t, dataset, model, device, memory_loader, all_probe_results, all_probe_train_results, train_stats, test_stats, end_task=False)

      ## BELOW for class-il and task-il evaluation, specific to supervised continual setting

      if args.cl_default:  
        if epoch > 0:
          old_fcs.pop(-1)
        old_fcs.append(deepcopy(get_head(model.net.backbone)))      
        accs = evaluate(model.net.backbone, dataset, device, debug=args.debug and args.debug_lpft)
        # results_mask_classes.append(accs[1])

        mean_acc = np.mean(accs, axis=1)
        if args.train.all_tasks_num_epochs > 0:
          # use the last fc each test task         
          old_fcs = [old_fcs[-1] for _ in range(dataset.N_TASKS)]
        task_accs = evaluate(model.net.backbone, dataset, device, fc=old_fcs, debug=args.debug and args.debug_lpft)

        mean_acc_task_il = np.mean(task_accs,axis=1)

        test_stats['class_il_mean_acc'].append(mean_acc[0])
        test_stats['task_il_mean_acc'].append(mean_acc_task_il[1])
        if not args.debug_lpft and "tune" in os.environ["logging"]: 
          tune.report(class_il_mean_acc=mean_acc[0])
          tune.report(task_il_mean_acc=mean_acc_task_il[1])
          for i in range(len(task_accs[1])):
            tune.report(**{f"class_il_acc_{i}": accs[1][i]})
            tune.report(**{f"task_il_acc_{i}": task_accs[1][i]})
        if not args.debug_lpft and "wandb" in os.environ["logging"]: 
          wandb.log({'class_il_mean_acc': mean_acc[0]})
          wandb.log({'task_il_mean_acc': mean_acc_task_il[1]})  
          for i in range(len(task_accs[1])):            
            wandb.log({f'class_il_acc_{i}': accs[0][i]})
            wandb.log({f'task_il_acc_{i}': task_accs[1][i]})
        if args.debug_lpft:
          print({'class_il_mean_acc': mean_acc[0]})
          print({'task_il_mean_acc': mean_acc_task_il[1]})
        # print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)

      logger.process_stats(train_stats, test_stats)
      train_metrics.append(train_stats)
      test_metrics.append(test_stats)
      logger.write_tsv(train_metrics, test_metrics)


    if not args.train.save_best:
      save_model(model,args,t,epoch,dataset)
    
    if hasattr(model, 'end_task'):
      model.end_task(dataset)





  if "tune" in os.environ["logging"]:
    tune.report(done=1)

  if args.eval is not False and args.cl_default is False:
      args.eval_from = model_path

  if "wandb" in os.environ["logging"]:
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
    
  

def init_wandb(args):
    wandb.init(project=args.project_name, name=args.run_name,
               group=args.group_name, config=vars(args.train))


def get_loss_acc_batch(images1, labels, images2, notaug_images, model, args):
    if args.cl_default:
        labels = labels.to(model.device)
        outputs = model.net.module.backbone(inputs1.to(model.device))
        loss = model.loss(outputs, labels).mean()
        # TODO: add accuracy numbers
        data_dict = {'loss': loss}
        data_dict['penalty'] = 0.0
    else:
        data_dict = model.net.forward(inputs1.to(model.device, non_blocking=True), inputs2.to(model.device, non_blocking=True))
        data_dict['loss'] = data_dict['loss'].mean()
        loss = data_dict['loss']
        data_dict['penalty'] = 0.0


def get_stats(loader, model, args, epoch):
    local_progress=tqdm(loader, desc=f'Epoch {epoch}/{args.train.num_epochs} (Testing)', disable=args.hide_progress)
    num_examples = 0
    num_correct = 0
    for idx, (inputs, labels) in enumerate(local_progress):
        outputs = model(inputs.to(model.device))
        _, predicted = torch.max(outputs, dim=1)
        predicted = predicted.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        if args.cl_default:
            num_correct += np.sum(predicted == labels)
        num_examples += len(inputs)
    return float(num_correct) / num_examples


# def main(device, args):
#     use_wandb=True
#     print(args.cl_default)
#     if args.train.num_epochs != args.train.stop_at_epoch:
#         print(f'warning: stop_at_epoch was {args.train.stop_at_epoch} and num_epochs was {args.train.num_epochs}')
#         print('\tbut making stop_at_epoch {args.train.num_epochs} now...')
#         args.train.stop_at_epoch = args.train.num_epochs
#     if use_wandb:
#         init_wandb(args)
#     dataset = get_dataset(args)
#     dataset_copy = get_dataset(args)
#     train_loader, memory_loader, test_loader = dataset_copy.get_data_loaders(args)

#     # define model
#     model = get_model(args, device, len(train_loader), dataset.get_transform(args))
#     logger = Logger(matplotlib=args.logger.matplotlib, log_dir=args.log_dir)
#     accuracy = 0 

#     if 'lp_epoch_frac' in args.train.__dict__:
#         print('Running lp-ft from task 1 onwards (for task 0 do full fine-tuning).')
#     for t in range(dataset.N_TASKS):
#         train_loader, memory_loader, test_loader = dataset.get_data_loaders(args)
#         global_progress = tqdm(range(0, args.train.stop_at_epoch), desc=f'Training')
#         for epoch in global_progress:
#             # TODO: for self-supervised methods, should we keep it on train.
#             # So check args.cl_default?
#             if t != 0 and 'lp_epoch_frac' in args.train.__dict__:
#                 if epoch == 0:
#                     print('Freezing model.')
#                     model.net.module.backbone.requires_grad_(False)
#                 if args.cl_default:
#                     model.net.module.backbone.fc.requires_grad_(True)     
#                 else:
#                     model.net.module.projector.requires_grad_(True)
#                 if epoch == int(args.train.stop_at_epoch * args.train.lp_epoch_frac):
#                     print('Unfreezing model.')
#                     model.net.module.backbone.requires_grad_(True)

#             stats = {}
#             model.eval()
#             test_acc = get_stats(test_loader, model, args, epoch)
#             print('test_acc: ', test_acc)
#             model.train()
#             results, results_mask_classes = [], []
#             total_loss = 0.0
#             num_correct = 0
#             num_examples = 0
#             local_progress=tqdm(train_loader, desc=f'Epoch {epoch}/{args.train.num_epochs}', disable=args.hide_progress)
#             for idx, ((images1, images2, notaug_images), labels) in enumerate(local_progress):
#                 data_dict = model.observe(images1, labels, images2, notaug_images)
#                 # TODO: add num_correct to model.observe.
#                 total_loss += data_dict['loss']
#                 num_examples += len(images1)
#                 if 'num_correct' in data_dict:
#                     num_correct += data_dict['num_correct']
#                 # TODO: figure out what logger does?
#                 logger.update_scalers(data_dict)
#             train_loss = total_loss / num_examples
#             stats['epoch'] = epoch
#             # stats['test_loss'] = test_loss
#             stats['train_loss'] = train_loss
#             if 'num_correct' in data_dict:
#                 stats['train_acc'] = float(num_correct) / num_examples
#                 stats['test_acc'] = test_acc
#             global_progress.set_postfix(data_dict)

#             if args.train.knn_monitor and epoch % args.train.knn_interval == 0: 
#                 for i in range(len(dataset.test_loaders)):
#                     acc, acc_mask = knn_monitor(model.net.module.backbone, dataset, dataset.memory_loaders[i], dataset.test_loaders[i], device, args.cl_default, task_id=t, k=min(args.train.knn_k, len(memory_loader.dataset))) 
#                     results.append(acc)
#                     stats[f'knn_acc_task_{i+1}'] = acc
#                 mean_acc = np.mean(results)
#                 stats['knn_mean_acc'] = mean_acc
         
#             if use_wandb:
#                 wandb.log(stats)
#             epoch_dict = {"epoch":epoch, "accuracy": mean_acc}
#             global_progress.set_postfix(epoch_dict)
#             logger.update_scalers(epoch_dict)
     
#         print("ending task")
#         if args.cl_default:
#             accs = evaluate(model.net.module.backbone, dataset, device)
#             mean_acc = np.mean(accs, axis=1)
#             print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)
 
#         model_path = os.path.join(args.ckpt_dir, f"{args.model.cl_model}_{t}.pth")
#         torch.save({
#             'epoch': epoch+1,
#             'state_dict':model.net.state_dict()
#         }, model_path)
#         print(f"Task Model saved to {model_path}")
#         with open(os.path.join(args.log_dir, f"checkpoint_path.txt"), 'w+') as f:
#             f.write(f'{model_path}')
      
#         if hasattr(model, 'end_task'):
#             model.end_task(dataset)
        
#         probe_train_results = []
#         probe_results = []
#         for i in range(len(dataset.test_loaders)):
#             train_acc, acc, best_c = logistic_monitor(model.net.module.backbone, dataset, dataset.memory_loaders[i], dataset.test_loaders[i], device, args.cl_default, task_id=t, k=min(args.train.knn_k, len(memory_loader.dataset))) 
#             probe_results.append(acc)
#             probe_train_results.append(train_acc)
#             if use_wandb:
#                 wandb.log({f"probe_acc_task_{i+1}": acc})
#                 wandb.log({f"probe_train_acc_task_{i+1}": train_acc})
#         mean_acc = np.mean(probe_results)
#         mean_train_acc = np.mean(probe_train_results)
#         if use_wandb:
#             wandb.log({f"probe_mean_acc": mean_acc})
#             wandb.log({f"probe_train_mean_acc": mean_train_acc})

#     # TODO: save all the probe results.
#     if args.eval is not False and args.cl_default is False:
#         args.eval_from = model_path

if __name__ == "__main__":
    args, checkpoints_dir = get_args()
    train(args=args)
    completed_log_dir = args.log_dir.replace('in-progress', 'debug' if args.debug else 'completed')
    os.rename(args.log_dir, completed_log_dir)
    if args.tmp_par_ckp_dir is not None:
      if args.save_model:
        new_checkpoints_dir = args.ckpt_dir
        shutil.copytree(checkpoints_dir, new_checkpoints_dir)
        print(f'Model has been moved to {new_checkpoints_dir}')
    print(f'Log file has been saved to {completed_log_dir}')

