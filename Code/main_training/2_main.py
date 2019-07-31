import torch
from torch.utils.data import DataLoader
import torchvison.transforms as transforms
import numpy as np
import os
from utils import MyDataset,validate,show_confMat
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
from tensorboadX import SummaryWriter
from datetime import datetime

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
writer = SummaryWriter()
# ------------data load ---------
# -------- pre pare data -----------
normMean = [0.4948052, 0.48568845, 0.44682974]
normStd = [0.24580306, 0.24236229, 0.2603115]
normalTransform = transforms.Normalize(normMean,normStd)
trainTransform = transforms.Compose([
    transforms.Resize(32),
    transforms.RandomCrop(32,padding=4),
    transforms.ToTensor(),
    normalTransform
])
validTransform = transforms.Compose([
    transforms.ToTensor(),
    normalTransform
])
#-----MyDataset-------
train_data = MyDataset(text_path = train_txt_path,transforms = trainTransform)
valid_data = MyDataset(text_path = valid_txt_path,transforms = validTransform)

# ------ net definition ------------
class Net (nn.Module):
    def __init__(self):
        super(Net,self).init()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.pool2 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
    def forward(self,x):
        x = self.pool1(F.relu(conv1(x)))
        x = self.pool2(F.relu(conv2(x)))
        x = self.view(-1,16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return x
# ---- initialization weight -------
    def initialization(self):
        for m in self.modules():
            if isinstance(m,nn.conv2):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
                elif isinstance(m,nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m,nn.Linear):
                    torch.nn.init.normal_(m.weight.data,0,0.01)
                    m.bias.data.zero_()
    net = Net()
    net.initialize_weights()
# ---- define lossfunc and optimizer---
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr = lr_init, momentum = 0.9, dampening = 0.1)
scheduler = torch.optim.lr.lr_scheduler.StepLR(optimizer,step_size = 50,gamma = 0.1)

# -----    train    ----------
for epoch in range(max_epoch):
    loss_sigma = 0.0
    correct = 0.0
    total = 0.0
    scheduler.step()

    for i, data in enumrate(train_loader):
        inputs, labels = train_data
        inputs, labels = Variable(inputs),Variable(labels)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs,lables)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data,1)
        total += lables.size(0)
        correct += (predicted == labels).squeeze().sum().numpy()
        loss_sigma += loss.item()

        if i%10 == 9:
           loss_avg = loss_sigma/10
           loss_sigma = 0
           print("Training epoch:[{:0>3}/{:0>3}] Iteration:[{:0>3}/{:0>3}]   Loss:[:.4f] Acc:{:.2%} ".format(epoch+1,max_epoch,i+1,len(train_loader),loss_avg,correct/total))

