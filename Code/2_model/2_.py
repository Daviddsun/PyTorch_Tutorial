import torch
from torch.utils.data import DataLoader
import torchvison.transform as transform
import numpy as np
import os
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys

train_txt_path = '../../Data/train.txt'
valid_txt_path = '../../Data/valid.txt'

classes_name = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

train_bs = 16
valid_bs = 16
lr_init = 0.001
max_epoch = 1

# ---- log ----
result_dir = '../../result/'
now_time = data.time.now()
time_str = data.time.strftime(now_time,'%m-%d-%H-%M-%S')

log_dir = os.path.join(result_dir,time_str)
if not exists(logdir):
    os.makedirs(log_dir)

# ------------data load ---------
normMean = [0.4948052, 0.48568845, 0.44682974]
normStd = [0.24580306, 0.24236229, 0.2603115]
