import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import numpy as np
from tqdm import tqdm
from arguments import get_args
from augmentations import get_aug
from models import get_model
from tools import AverageMeter, knn_monitor, logistic_monitor, Logger, file_exist_check
from datasets import get_dataset
from datetime import datetime
from utils.loggers import *
from utils.metrics import mask_classes
from utils.loggers import CsvLogger
from datasets.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel
from typing import Tuple
import wandb


def evaluate(model: ContinualModel, dataset: ContinualDataset, device, classifier=None) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.training
    model.eval()
    accs, accs_mask_classes = [], []
    for k, test_loader in enumerate(dataset.test_loaders):
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if classifier is not None:
                outputs = classifier(outputs)

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


def main(device, args):
    use_wandb=True
    print(args.cl_default)
    if args.train.num_epochs != args.train.stop_at_epoch:
        print(f'warning: stop_at_epoch was {args.train.stop_at_epoch} and num_epochs was {args.train.num_epochs}')
        print('\tbut making stop_at_epoch {args.train.num_epochs} now...')
        args.train.stop_at_epoch = args.train.num_epochs
    if use_wandb:
        init_wandb(args)
    dataset = get_dataset(args)
    dataset_copy = get_dataset(args)
    train_loader, memory_loader, test_loader = dataset_copy.get_data_loaders(args)

    # define model
    model = get_model(args, device, len(train_loader), dataset.get_transform(args))
    logger = Logger(matplotlib=args.logger.matplotlib, log_dir=args.log_dir)
    accuracy = 0 

    if 'lp_epoch_frac' in args.train.__dict__:
        print('Running lp-ft from task 1 onwards (for task 0 do full fine-tuning).')
    for t in range(dataset.N_TASKS):
        train_loader, memory_loader, test_loader = dataset.get_data_loaders(args)
        global_progress = tqdm(range(0, args.train.stop_at_epoch), desc=f'Training')
        for epoch in global_progress:
            # TODO: for self-supervised methods, should we keep it on train.
            # So check args.cl_default?
            if t != 0 and 'lp_epoch_frac' in args.train.__dict__:
                if epoch == 0:
                    print('Freezing model.')
                    model.net.module.backbone.requires_grad_(False)
                if args.cl_default:
                    model.net.module.backbone.fc.requires_grad_(True)     
                else:
                    model.net.module.projector.requires_grad_(True)
                if epoch == int(args.train.stop_at_epoch * args.train.lp_epoch_frac):
                    print('Unfreezing model.')
                    model.net.module.backbone.requires_grad_(True)

            stats = {}
            model.eval()
            test_acc = get_stats(test_loader, model, args, epoch)
            print('test_acc: ', test_acc)
            model.train()
            results, results_mask_classes = [], []
            total_loss = 0.0
            num_correct = 0
            num_examples = 0
            local_progress=tqdm(train_loader, desc=f'Epoch {epoch}/{args.train.num_epochs}', disable=args.hide_progress)
            for idx, ((images1, images2, notaug_images), labels) in enumerate(local_progress):
                data_dict = model.observe(images1, labels, images2, notaug_images)
                # TODO: add num_correct to model.observe.
                total_loss += data_dict['loss']
                num_examples += len(images1)
                if 'num_correct' in data_dict:
                    num_correct += data_dict['num_correct']
                # TODO: figure out what logger does?
                logger.update_scalers(data_dict)
            train_loss = total_loss / num_examples
            stats['epoch'] = epoch
            # stats['test_loss'] = test_loss
            stats['train_loss'] = train_loss
            if 'num_correct' in data_dict:
                stats['train_acc'] = float(num_correct) / num_examples
                stats['test_acc'] = test_acc
            global_progress.set_postfix(data_dict)

            if args.train.knn_monitor and epoch % args.train.knn_interval == 0: 
                for i in range(len(dataset.test_loaders)):
                    acc, acc_mask = knn_monitor(model.net.module.backbone, dataset, dataset.memory_loaders[i], dataset.test_loaders[i], device, args.cl_default, task_id=t, k=min(args.train.knn_k, len(memory_loader.dataset))) 
                    results.append(acc)
                    stats[f'knn_acc_task_{i+1}'] = acc
                mean_acc = np.mean(results)
                stats['knn_mean_acc'] = mean_acc
         
            if use_wandb:
                wandb.log(stats)
            epoch_dict = {"epoch":epoch, "accuracy": mean_acc}
            global_progress.set_postfix(epoch_dict)
            logger.update_scalers(epoch_dict)
     
        print("ending task")
        if args.cl_default:
            accs = evaluate(model.net.module.backbone, dataset, device)
            mean_acc = np.mean(accs, axis=1)
            print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)
 
        model_path = os.path.join(args.ckpt_dir, f"{args.model.cl_model}_{t}.pth")
        torch.save({
            'epoch': epoch+1,
            'state_dict':model.net.state_dict()
        }, model_path)
        print(f"Task Model saved to {model_path}")
        with open(os.path.join(args.log_dir, f"checkpoint_path.txt"), 'w+') as f:
            f.write(f'{model_path}')
      
        if hasattr(model, 'end_task'):
            model.end_task(dataset)
        
        probe_train_results = []
        probe_results = []
        for i in range(len(dataset.test_loaders)):
            train_acc, acc, best_c = logistic_monitor(model.net.module.backbone, dataset, dataset.memory_loaders[i], dataset.test_loaders[i], device, args.cl_default, task_id=t, k=min(args.train.knn_k, len(memory_loader.dataset))) 
            probe_results.append(acc)
            probe_train_results.append(train_acc)
            if use_wandb:
                wandb.log({f"probe_acc_task_{i+1}": acc})
                wandb.log({f"probe_train_acc_task_{i+1}": train_acc})
        mean_acc = np.mean(probe_results)
        mean_train_acc = np.mean(probe_train_results)
        if use_wandb:
            wandb.log({f"probe_mean_acc": mean_acc})
            wandb.log({f"probe_train_mean_acc": mean_train_acc})

    # TODO: save all the probe results.
    if args.eval is not False and args.cl_default is False:
        args.eval_from = model_path

if __name__ == "__main__":
    args = get_args()
    main(device=args.device, args=args)
    completed_log_dir = args.log_dir.replace('in-progress', 'debug' if args.debug else 'completed')
    os.rename(args.log_dir, completed_log_dir)
    print(f'Log file has been saved to {completed_log_dir}')

