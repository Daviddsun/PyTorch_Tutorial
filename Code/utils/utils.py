# coding: utf-8
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
from torch.autograd import Variable
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    # 定义权值初始化
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()

class FocalBinaryCrossEntropy(nn.Module):
    """
        As title
    """

    def __init__(self, size_average=True, ignore_label=-1, gamma=0, alpha=0.5):
        """
            Args:
                size_average
                ignore_label
        """
        super(FocalBinaryCrossEntropy, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

        self.gamma = gamma
        self.alpha = alpha

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        # assert predict.dim() == 3
        # assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        # assert predict.size(1) == target.size(1), "{0} vs {1} ".format(predict.size(1), target.size(1))
        # assert predict.size(2) == target.size(2), "{0} vs {1} ".format(predict.size(2), target.size(2))
        target_mask = (target != self.ignore_label)
        target = target[target_mask]

        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict[target_mask]

        predict = predict.view(-1, 1)
        target = target.view(-1, 1)

        pt = torch.sigmoid(predict)
        logpt = torch.log(pt)

        npt = 1 - torch.sigmoid(predict)
        lognpt = torch.log(npt)

        loss_positive = -1 * self.alpha * target * logpt * (1 - pt) ** self.gamma
        loss_negtive = -1 * (1 - self.alpha) * (1 - target) * lognpt * 0.5

        focal_loss = (loss_positive + loss_negtive).mean()
        return focal_loss, loss_positive.mean(), loss_negtive.mean()


class MyDataset(Dataset):
    def __init__(self, txt_path, transform = None, target_transform = None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))

        self.imgs = imgs        # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')     # 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1

        if self.transform is not None:
            img = self.transform(img)   # 在这里做transform，转为tensor等等

        return img, label

    def __len__(self):
        return len(self.imgs)


def validate(net, data_loader, set_name, classes_name):
    """
    对一批数据进行预测，返回混淆矩阵以及Accuracy
    :param net:
    :param data_loader:
    :param set_name:  eg: 'valid' 'train' 'tesst
    :param classes_name:
    :return:
    """
    net.eval()
    cls_num = len(classes_name)
    conf_mat = np.zeros([cls_num, cls_num])

    for data in data_loader:
        images, labels = data
        images = Variable(images)
        labels = Variable(labels)

        outputs = net(images)
        outputs.detach_()

        _, predicted = torch.max(outputs.data, 1)

        # 统计混淆矩阵
        for i in range(len(labels)):
            cate_i = labels[i].numpy()
            pre_i = predicted[i].numpy()
            conf_mat[cate_i, pre_i] += 1.0

    for i in range(cls_num):
        print('class:{:<10}, total num:{:<6}, correct num:{:<5}  Recall: {:.2%} Precision: {:.2%}'.format(
            classes_name[i], np.sum(conf_mat[i, :]), conf_mat[i, i], conf_mat[i, i] / (1 + np.sum(conf_mat[i, :])),
                                                                conf_mat[i, i] / (1 + np.sum(conf_mat[:, i]))))

    print('{} set Accuracy:{:.2%}'.format(set_name, np.trace(conf_mat) / np.sum(conf_mat)))

    return conf_mat, '{:.2}'.format(np.trace(conf_mat) / np.sum(conf_mat))


def show_confMat(confusion_mat, classes, set_name, out_dir):

    # 归一化
    confusion_mat_N = confusion_mat.copy()
    for i in range(len(classes)):
        confusion_mat_N[i, :] = confusion_mat[i, :] / confusion_mat[i, :].sum()

    # 获取颜色
    cmap = plt.cm.get_cmap('Greys')  # 更多颜色: http://matplotlib.org/examples/color/colormaps_reference.html
    plt.imshow(confusion_mat_N, cmap=cmap)
    plt.colorbar()

    # 设置文字
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, list(classes), rotation=60)
    plt.yticks(xlocations, list(classes))
    plt.xlabel('Predict label')
    plt.ylabel('True label')
    plt.title('Confusion_Matrix_' + set_name)

    # 打印数字
    for i in range(confusion_mat_N.shape[0]):
        for j in range(confusion_mat_N.shape[1]):
            plt.text(x=j, y=i, s=int(confusion_mat[i, j]), va='center', ha='center', color='red', fontsize=10)
    # 保存
    plt.savefig(os.path.join(out_dir, 'Confusion_Matrix' + set_name + '.png'))
    plt.close()


def normalize_invert(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor
