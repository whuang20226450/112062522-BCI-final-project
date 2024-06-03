import random
import logging
from pprint import pformat
from argparse import ArgumentParser
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import time
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import shutil

import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import functional as F
from torchvision.transforms import functional as tf
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from torchvision import models
from torchvision import transforms as trans
from torchsummary import summary
from lion_pytorch import Lion

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from data import create_loaderHandler
from net import create_net
from loss import create_loss

class EEGNetModel:
  def __init__(self, conf):
    # config file from yaml and setting
    self.conf = conf
    self.device = self.conf['device']
    
    # Output Setting
    self.workspace_path = Path(self.conf['workspace_root_path']) / self.conf['name']
    self.workspace_weights_path = self.workspace_path / "weights"
    
    
  ###
  # Train Sections
  ###
  
  
  def train_init(self):
    logging.info("Model Train Init")
    
    self.device = self.conf['device']
    
    ## datasets loader
    self.dataloader_handler = create_loaderHandler(self.conf, mode='train')
    self.train_loader = self.dataloader_handler.train_loader
    self.val_loader   = self.dataloader_handler.val_loader
    
    # Training Setting
    self.max_epoch = self.conf['train_setting']['max_epoch']
    self.init_lr = float(self.conf['train_setting']['init_lr'])
    self.max_lr = float(self.conf['train_setting']['max_lr'])
    self.warmup_epoch = self.conf['train_setting']['warmup_epoch']
    self.weight_decay = float(self.conf['train_setting']['weight_decay'])
    self.save_weight_period = self.conf['train_setting']['save_weight_period']
    self.num_classes = self.conf['train_setting']['num_classes']
    self.num_subjects = self.conf['train_setting']['num_subjects']
    self.num_samples = self.conf['train_setting']['num_samples']
    self.num_channels = self.conf['train_setting']['num_channels']
    
    self.seed = 123
    self.set_seed(self.seed)
    
    logging.info(f"max_epoch: {self.max_epoch}")
    logging.info(f"init_lr: {self.init_lr}")
    logging.info(f"weight_decay: {self.weight_decay}")
    logging.info(f"seed: {self.seed}")
    
    self.net = create_net(self.conf)
    self.criterion = create_loss(self.conf)
    # self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
    self.optimizer = Lion(self.net.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
    if self.warmup_epoch > 0:
      ratio = self.max_lr / self.init_lr
      lambda1 = lambda epoch: ratio**(epoch / self.warmup_epoch) if epoch < self.warmup_epoch else ratio
      self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda1)

  
  def load_resume_weight(self, resume_weigth_file_path):
    '''
    Load the checkpoint into net and optimizer
    Note: please run train_init() first
    '''
    # checkpoint = torch.load(resume_weigth_file_path)
    # self.net.load_state_dict(checkpoint['model_state_dict'])
    # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # logging.info(f"Load weight success: {str(resume_weigth_file_path)}")
    
    pass
  
  
  def run_train(self):
    '''
      Model Training Functions
    '''
    self.metrics = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'lr': [], 'train_macro_f1': [], 'val_macro_f1': [], 'val_micro_f1': []}
    self.scaler = torch.cuda.amp.GradScaler()
    self.total, self.max_acc, self.max_macrof1 = 0, 0, 0
    
    for self.epoch in range(self.max_epoch):
      start_time = time.time()
      
      logging.info(f"Epoch: {self.epoch} Start")
      self.train()
      self.validation()
      if self.warmup_epoch > 0:
        self.scheduler.step()

      duration = time.time() - start_time
      
      if self.warmup_epoch > 0:
        logging.info(f"epoch {self.epoch}: val acc: {self.metrics['val_acc'][-1]} / loss: {self.metrics['val_loss'][-1]} / lr: {self.scheduler.get_last_lr()} / duration: {duration:2f}")
        self.metrics['lr'].append(self.scheduler.get_last_lr())
      else:
        logging.info(f"epoch {self.epoch}: val acc: {self.metrics['val_acc'][-1]} / loss: {self.metrics['val_loss'][-1]} / duration: {duration:2f}")
      logging.info(f"epoch {self.epoch}: val micro f1: {self.metrics['val_micro_f1'][-1]}")
      logging.info(f"epoch {self.epoch}: val macro f1: {self.metrics['val_macro_f1'][-1]}")
      logging.info(f"epoch {self.epoch}: best val macro f1: {self.max_macrof1} / best val acc: {self.max_acc}")
        
    self.plot_training()
  
  def train(self):
    '''
      training function running in run_train()
    '''
    self.net.train()
    
    running_loss = 0.0
    count = 0
    record_y = torch.zeros(1)
    record_y_pred = torch.zeros(1, self.num_classes)
    
    for i, data in enumerate(self.train_loader):
      time_pin = time.time()
       
      x = data['x'].to(self.device)
      y = data['y'].to(self.device).squeeze(-1)
      x = x.view(-1, 1, self.num_channels, self.num_samples)
      
      with torch.cuda.amp.autocast():
        y_pred = self.net(x)
        loss = self.criterion(y_pred, y)
      
      record_y = torch.cat((record_y, y.detach().cpu()), 0)
      record_y_pred = torch.cat((record_y_pred, y_pred.detach().cpu()), 0)
        
      self.scaler.scale(loss).backward()
      self.scaler.step(self.optimizer)
      self.scaler.update()
      self.optimizer.zero_grad()
      
      cur_batchsize = x.size(0)
      running_loss += loss.item() * cur_batchsize
      count += cur_batchsize
      
      if count % 10000 == 0:
        if self.total == 0:
          logging.info(f"epoch {self.epoch}: {count}/unknown finished / train loss: {running_loss / count:.4f} / duration: {time.time() - time_pin:2f}")
        else:
          logging.info(f"epoch {self.epoch}: {count}/{self.total} finished / train loss: {running_loss / count:.4f} / duration: {time.time() - time_pin:2f}")

      # if self.steps % self.save_weight_period == 0:
      #   logging.info(f"Now Steps: {self.steps}")
      #   logging.info(f"Periodically saving weight")
      #   torch.save({
      #       'model_state_dict': self.net.state_dict(),
      #       'optimizer_state_dict': self.optimizer.state_dict(),
      #   }, self.workspace_weights_path / f"model_{self.steps}.pt")
      
    self.total = count
    y_true = record_y[1::].cpu().detach().numpy()
    y_pred = record_y_pred[1::].cpu().detach().numpy()
    y_pred = np.argmax(y_pred, axis=1)
    acc = accuracy_score(y_true, y_pred)
    macro_score = f1_score(y_true, y_pred, average='macro')
    self.metrics['train_acc'].append(acc)
    self.metrics['train_loss'].append(running_loss / self.total)
    self.metrics['train_macro_f1'].append(macro_score)
    
    
  @torch.no_grad()
  def validation(self):
    self.net.eval()
    acc, macro_f1, micro_f1, loss, total, y_true, y_pred = self.evaluate(self.net, self.val_loader)
    
    self.metrics['val_acc'].append(acc)
    self.metrics['val_loss'].append(loss / total)
    self.metrics['val_macro_f1'].append(macro_f1)
    self.metrics['val_micro_f1'].append(micro_f1)
    
    if acc > self.max_acc:
        self.max_acc = acc
        torch.save({
          'model_state_dict': self.net.state_dict(),
          'optimizer_state_dict': self.optimizer.state_dict(),
        }, self.workspace_weights_path / f"model_best_acc.pt")
        
        np.save(f'{self.workspace_path}/results_acc.npy', {'y_true': y_true, 'y_pred': y_pred})
        
    if macro_f1 > self.max_macrof1:
        self.max_macrof1 = macro_f1
        torch.save({
          'model_state_dict': self.net.state_dict(),
          'optimizer_state_dict': self.optimizer.state_dict(),
        }, self.workspace_weights_path / f"model_best_macrof1.pt")
        
        np.save(f'{self.workspace_path}/results_macrof1.npy', {'y_true': y_true, 'y_pred': y_pred})
    
    return acc, macro_f1, micro_f1, loss, total
  
  
  def print_net(self):
    '''
    print out the net arch.
    '''
    # imgs, labels, dicom_ids = next(iter(self.train_loader))
    # logging.info(f"img.shape: {imgs.shape}")
    # logging.info(f"labels.shape: {labels.shape}")
    # logging.info(f"dicom_ids.shape: {dicom_ids.shape}")
    # logging.info(f"\n {self.net}")
    
    # summary(self.net, (3, 256, 256))
    pass
    
  ###
  # Test
  ###

  def test_init(self):
    logging.info("Model Test Init")
      
    # Training Setting
    self.weight_path = self.conf['weight_path']
    self.max_epoch = self.conf['train_setting']['max_epoch']
    self.init_lr = float(self.conf['train_setting']['init_lr'])
    self.max_lr = float(self.conf['train_setting']['max_lr'])
    self.warmup_epoch = self.conf['train_setting']['warmup_epoch']
    self.weight_decay = float(self.conf['train_setting']['weight_decay'])
    self.save_weight_period = self.conf['train_setting']['save_weight_period']
    self.num_classes = self.conf['train_setting']['num_classes']
    self.num_subjects = self.conf['train_setting']['num_subjects']
    self.num_samples = self.conf['train_setting']['num_samples']
    self.num_channels = self.conf['train_setting']['num_channels']
    
    logging.info(f"Model Load Weight: {str(self.weight_path)}")
    self.net = create_net(self.conf)
    self.net.load_state_dict(torch.load(str(self.weight_path))["model_state_dict"])    
    self.criterion = create_loss(self.conf)
    
    ## datasets loader
    self.dataloader_handler = create_loaderHandler(self.conf, mode='test')
    self.test_loader = self.dataloader_handler.test_loader


  def run_test(self):
    self.net.eval()
    acc, macro_f1, micro_f1, loss, total, y_true, y_pred = self.evaluate(self.net, self.test_loader)

    logging.info(f"Test Accuracy: {acc}")
    logging.info(f"Test Macro F1: {macro_f1}")
    logging.info(f"Test Micro F1: {micro_f1}")
    self.plot_testing(y_true, y_pred)
    
    return 


  
  ###
  ## Utiles
  ##
  

  def set_seed(self, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


  @torch.no_grad()
  def evaluate(self, model, loader):
    model.eval()
    running_loss, total = 0., 0

    with torch.no_grad():
      record_target_label = torch.zeros(1)
      record_predict_label = torch.zeros(1, self.num_classes)

      for i, data in enumerate(loader):
        x = data['x'].to(self.device)
        y = data['y'].to(self.device).squeeze(-1)
        x = x.view(-1, 1, self.num_channels, self.num_samples)

        y_pred = model(x)
        loss = self.criterion(y_pred, y)

        cur_bs = x.size(0)
        running_loss += loss.item() * cur_bs
        total += cur_bs

        record_target_label = torch.cat((record_target_label, y.detach().cpu()), 0)
        record_predict_label = torch.cat((record_predict_label, y_pred.detach().cpu()), 0)

      y_true = record_target_label[1::].cpu().detach().numpy()
      y_pred = record_predict_label[1::].cpu().detach().numpy()
      y_pred = np.argmax(y_pred, axis=1)

      accuracy = accuracy_score(y_true, y_pred)

      macro_score = f1_score(y_true, y_pred, average='macro')
      micro_score = f1_score(y_true, y_pred, average='micro')
            
    return accuracy, macro_score, micro_score, running_loss, total, y_true, y_pred

  def plot_training(self):
    # Create a new figure and a subplot with its own y-axis
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot training and validation accuracy on the first y-axis (left)
    ax1.plot(self.metrics['train_acc'], 'b-', label='Training Accuracy')  # Solid blue line
    ax1.plot(self.metrics['val_acc'], 'b--', label='Validation Accuracy')  # Dashed blue line
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy', color='b')
    ax1.tick_params('y', colors='b')

    # Create a second y-axis for the same x-axis
    ax2 = ax1.twinx()

    # Plot training and validation loss on the second y-axis (right)
    ax2.plot(self.metrics['train_loss'], 'r-', label='Training Loss')  # Solid red line
    ax2.plot(self.metrics['val_loss'], 'r--', label='Validation Loss')  # Dashed red line
    ax2.set_ylabel('Loss', color='r')
    ax2.tick_params('y', colors='r')

    # Adding title and customizing layout
    plt.title('Training/Validation Accuracy and Loss')

    # Adding legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')

    plt.savefig(f'{self.workspace_path}/training.png', dpi=300)
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(self.metrics['lr'])
    plt.title('Learning Rate')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.savefig(f'{self.workspace_path}/learning_rate.png', dpi=300)
    plt.close()
    
    # ---------------
    # Plot f1 score
    plt.figure(figsize=(10, 6))
    plt.plot(self.metrics['train_macro_f1'], label='Training Macro F1')
    plt.plot(self.metrics['val_macro_f1'], label='Validation Macro F1')
    plt.title('Training/Validation Macro F1')
    plt.xlabel('Epochs')
    plt.ylabel('Macro F1')
    plt.legend()
    plt.savefig(f'{self.workspace_path}/f1_score.png', dpi=300)
    plt.close()
    
    # ---------------
    # Plot confusion matrix - based on best accuracy
    data = np.load(f'{self.workspace_path}/results_acc.npy', allow_pickle=True).item()
    y_true, y_pred = data['y_true'], data['y_pred']
    print(y_true.shape, y_pred.shape)
        
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues', cbar=False)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    plt.savefig(f'{self.workspace_path}/best_val_acc_confusionmatrix.png', dpi=300)
    plt.close()
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(cm_normalized, annot=True, ax=ax, square=True, cmap="Blues", fmt=".2f")
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    plt.savefig(f'{self.workspace_path}/noramlized_best_val_acc_confusionmatrix.png', dpi=300)
    plt.close()
    
    # plot confusion matrix - based on best macro f1
    data = np.load(f'{self.workspace_path}/results_macrof1.npy', allow_pickle=True).item()
    y_true, y_pred = data['y_true'], data['y_pred']
    
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues', cbar=False)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    plt.savefig(f'{self.workspace_path}/best_val_macrof1_confusionmatrix.png', dpi=300)
    plt.close()
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(cm_normalized, annot=True, ax=ax, square=True, cmap="Blues", fmt=".2f")
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    plt.savefig(f'{self.workspace_path}/noramlized_best_val_macrof1_confusionmatrix.png', dpi=300)
    plt.close()


  def plot_testing(self, y_true, y_pred):
    # plot confusion matrix - based on best macro f1
    
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues', cbar=False)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    plt.savefig(f'{self.workspace_path}/best_test_macrof1_confusionmatrix.png', dpi=300)
    plt.close()
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(cm_normalized, annot=True, ax=ax, square=True, cmap="Blues", fmt=".2f")
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    plt.savefig(f'{self.workspace_path}/noramlized_best_test_macrof1_confusionmatrix.png', dpi=300)
    plt.close()
