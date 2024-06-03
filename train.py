import random
import logging
from pprint import pformat
from argparse import ArgumentParser
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import shutil

import torch
from torch import nn
import torch.optim as optim
from torch.nn import functional as F
from torchvision.transforms import functional as tf
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from torchvision import models



## Model
from model import create_model


###
## Utiles
###
def bootstrap():
  '''
    For initing the train program, make everything works find before staring
  '''
  ## init vars
  now_time = datetime.now()
  
  
  ## Parse args from cli args
  argParser = ArgumentParser()
  argParser.add_argument("cmd", help="what cmd you want to run, options: start / dev")
  argParser.add_argument("-c", "--config", help="yaml config file (usually in ./config folder)", dest="conf", required=True)
  args = argParser.parse_args()

  cmd = args.cmd
  
  ## yaml config file parse
  conf_file_path = args.conf # config file
  with open(conf_file_path, "r") as file:
    conf = yaml.safe_load(file) # conf file is loaded
  
  ## create experiments folder
  check_workspace_root_folder(conf) # check the workspace root is exits
  workspace_path = Path(conf['workspace_root_path']) / conf['name']
  workspace_weights_path = workspace_path / "weights"
  resume_weight_file_path = None

  # check if the workspace is exists
  if workspace_path.exists():
    # check if the resume state (weight) exists
    weights_path_list = sorted(list(workspace_weights_path.iterdir()))
    weights_filename_list = [weight_path.stem for weight_path in weights_path_list]
    
    # if weights_path_list have weight file in it
    if len(weights_path_list) > 0:
      select_idx = None
      max_step_num = -1

      # if the filename have "best" we chose it, else chose largest step pt files
      for idx, filename in enumerate(weights_filename_list):
        if "best" in filename:
          select_idx = idx
          break
        else:
          step_num = int(filename.split('_')[1])
          if step_num > max_step_num:
            max_step_num = step_num
            select_idx = idx
            
      resume_weight_file_path = weights_path_list[select_idx]
    else:
      resume_weight_file_path = None
    
    # move old folder to archived (if the conf['name'] is the existed in "workspace_root_path")
    workspace_archived_path = Path(conf['workspace_root_path']) / 'archived' / f"{conf['name']}_{now_time.strftime('%Y%m%d_%H%M%S')}"
    shutil.move(workspace_path, workspace_archived_path)
    
  # create workspace and models weights result
  workspace_path.mkdir(parents=False, exist_ok=False)
  workspace_weights_path.mkdir(parents=False, exist_ok=False)
  
  # move weight file if have it (only best or latest)
  if resume_weight_file_path is not None:
    # get moved it archived folder's old weight files path
    resume_weigth_old_file_path = workspace_archived_path / "weights" / resume_weight_file_path.name
    resume_weight_file_path = workspace_weights_path / f"resume-{resume_weight_file_path.name}"
    shutil.copy(resume_weigth_old_file_path, resume_weight_file_path)
  
  ## logger init
  FORMAT = '[%(asctime)s %(filename)s][%(levelname)s]: %(message)s'
  log_filename = f"{now_time.strftime('%Y%m%d_%H%M%S')}_{conf['name']}.log"
  log_path = workspace_path / log_filename
  logging.basicConfig(level=logging.INFO, format=FORMAT, 
                      handlers=[
                          logging.FileHandler(log_path, mode="a"),
                          logging.StreamHandler()
                        ]
                      )
  
  logging.info(f"Config File: \n {yaml.dump(conf, default_flow_style=False)}")
  
  return conf, cmd, resume_weight_file_path


def check_workspace_root_folder(conf):
  '''
    Check the workspace root folder is exists
  '''
  # check root folder is exist, if not than create it
  workspace_root_path = Path(conf['workspace_root_path'])
  if not workspace_root_path.exists():
    workspace_root_path.mkdir(parents=True, exist_ok=False)

  # check archived folder is exist, if not than create it
  workspace_root_archived_path = workspace_root_path / 'archived'
  if not workspace_root_archived_path.exists():
    workspace_root_archived_path.mkdir(parents=False, exist_ok=False)


###
## Trainer
###
class Trainer:
  def __init__(self, conf, resume_weight_file_path):
    ## config file from yaml and setting
    self.conf = conf
    self.device = self.conf['device']
    self.resume_weight_file_path = resume_weight_file_path
    
    ## Model Setting
    self.model = create_model(self.conf)
    
    # if self.resume_weight_file_path is not None:
    #   self.model.load_resume_weight(self.resume_weight_file_path)
    
  
  def run(self):
    self.model.train_init()
    logging.info("Model Training Start!")
    self.model.run_train()
    
    
  def run_dev(self):
    logging.info("Model Dev. Mode")
    self.model.print_net()
    
    
  def run_test(self):
    self.model.test_init()
    logging.info("Model Test Mode")
    self.model.run_test()
    

if __name__ == "__main__":
  # Bootstrap the train program
  conf, cmd, resume_weight_file_path = bootstrap()

  # Run trainer
  trainer = Trainer(conf, resume_weight_file_path)
  
  if cmd == "start":
    trainer.run()
    with open("experiments/result.txt", "a") as file:
      file.write(f"{conf['name']}: best val_acc: {trainer.model.max_acc} | best val_macrof1: {trainer.model.max_macrof1}\n")
  elif cmd == "test":
    trainer.run_test()
  else:
    logging.critical(f"Your cmd: {cmd} is not valid, look help docs.")

    
