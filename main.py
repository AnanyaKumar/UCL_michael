import os
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
from tools import AverageMeter, knn_monitor, Logger, file_exist_check
from datasets import get_dataset
from datetime import datetime
from utils.loggers import *
from utils.metrics import mask_classes
from utils.loggers import CsvLogger
from datasets.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel
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



def trainable(config):
  # WANDB    
  
  args = config["default_args"]
  device = args["device"]
  for (k, v) in config['train'].items(): 
    args['train'] = update_args(args['train'], k, v)

  os.environ['logging'] = "wandb,tune"
  if not args['debug_lpft']:
    wandb.init(project="lpft", config=args)
    
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
    train_loader, memory_loader, test_loader = dataset.get_data_loaders(args)
    if args.last and t < 4: 
      print("continuing cause only train last task...")
      continue

    global_progress = tqdm(range(0, args.train.stop_at_epoch), desc=f'Training')
    for epoch in global_progress:   
      if args.lpft and (not args.train.ft_first or t):
        if epoch == 0:
          model.net.module.backbone.requires_grad_(False)          
          if args.train.reset_lp_lr:
            for pg in model.opt.param_groups:
              pg['lr'] = args.train.lp_lr*args.train.batch_size/256
          if args.cl_default:
            model.net.module.backbone.fc.requires_grad_(True)    
          elif args.train.proj_is_head:
            model.net.module.projector.requires_grad_(False)      
        if epoch == args.train.num_lp_epochs:
          model.net.module.backbone.requires_grad_(True)          
          for pg in model.opt.param_groups:
            pg['lr'] = args.train.ft_lr*args.train.batch_size/256
          if not args.cl_default:
            model.net.module.projector.requires_grad_(True)

      if not args.train.train_first or not t:
        model.train()
      else:
        model.eval()
      results, results_mask_classes = [], []
      
      local_progress=tqdm(train_loader, desc=f'Epoch {epoch}/{args.train.num_epochs}', disable=args.hide_progress)
      t_1 = time.time()

      t__0 = time.time()
      loading_time = 0.
      observe_time = 0.
      for idx, ((images1, images2, notaug_images), labels) in enumerate(local_progress):
          # print("loading took", time.time()-t__0, "seconds")
          loading_time += (time.time()-t__0)
          t__0 = time.time()
          data_dict = model.observe(images1, labels, images2, notaug_images)
          # print("observing took", time.time()-t__0, "seconds"); t__0 = time.time()
          # logger.update_scalers(data_dict)
          if not args.debug_lpft and "tune" in os.environ["logging"]: tune.report(loss=data_dict['loss'].item())
          if not args.debug_lpft and "wandb" in os.environ["logging"]: wandb.log({'loss': data_dict['loss'].item()})
          # print("logger took", time.time()-t__0, "seconds")            
          observe_time += (time.time()-t__0)
          t__0 = time.time()

      # print("loading took", loading_time, "seconds")
      # print("observing took", observe_time, "seconds")

      t_2 = time.time()
      # print("train took", t_2-t_1, "seconds")
      global_progress.set_postfix(data_dict)

      if args.train.knn_monitor and epoch % args.train.knn_interval == 0: 
          for i in range(len(dataset.test_loaders)):
            
            acc, acc_mask = knn_monitor(model.net.module.backbone, dataset, dataset.memory_loaders[i], dataset.test_loaders[i], device, args.cl_default, task_id=t, k=min(args.train.knn_k, len(memory_loader.dataset))) 
            results.append(acc)
            if not args.debug_lpft and "tune" in os.environ["logging"]: tune.report(**{f"knn_acc_task_{i+1}": acc})
            if not args.debug_lpft and "wandb" in os.environ["logging"]: wandb.log({f"knn_acc_task_{i+1}": acc})
          mean_acc = np.mean(results)
          if not args.debug_lpft and "tune" in os.environ["logging"]: tune.report(**{f"knn_mean_acc": mean_acc})
          if not args.debug_lpft and "wandb" in os.environ["logging"]: wandb.log({f"knn_mean_acc": mean_acc})

          t_3 = time.time()
          # print("knn took", t_3-t_2, "seconds")
        
      epoch_dict = {"epoch":epoch, "accuracy": mean_acc}
      print("mean_accuracy:", mean_acc)
      global_progress.set_postfix(epoch_dict)
      logger.update_scalers(epoch_dict)
    
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
      # print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)

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
    t_4 = time.time()
    print("model save took", t_4-t_3, "seconds")
    with open(os.path.join(args.log_dir, f"checkpoint_path.txt"), 'w+') as f:
      f.write(f'{model_path}')
    
    if hasattr(model, 'end_task'):
      model.end_task(dataset)

  if args.eval is not False and args.cl_default is False:
      args.eval_from = model_path


def train(args):  
  config = {"default_args": vars(args), "train": {
    # "cl_default": tune.grid_search([True]),
    # "warmup_epochs": tune.grid_search([10]),
    # "warmup_lr": tune.grid_search([0]),
    # "lp_lr": tune.grid_search([0.03]),
    "ft_lr": tune.grid_search([0.03]),
    "num_lp_epochs": tune.grid_search([25]),
    # "proj_is_head": tune.grid_search([False]),
  }}
    ## RAY TUNE
  # tune.run(trainable, config=config, num_samples=1, resources_per_trial={"cpu": 15, "gpu": 1})
  if args.debug_lpft:
    trainable(config=config)
  else:
    tune.run(trainable, config=config, num_samples=1, resources_per_trial={"cpu": 16, "gpu": 1})
  


if __name__ == "__main__":
    args = get_args()
    train(args=args)
    completed_log_dir = args.log_dir.replace('in-progress', 'debug' if args.debug else 'completed')
    os.rename(args.log_dir, completed_log_dir)
    print(f'Log file has been saved to {completed_log_dir}')


