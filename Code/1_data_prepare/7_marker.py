# coding:utf-8
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import os
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
# from utils.utils import MyDataset
from PIL import Image


_7_marker_img = '../../7_marker/'
_7_marker_txt = '../../7_marker/7_marker.txt'
_7_marker_trans = '../../7_marker/trans/'

img_h, img_w = 320, 320
imgs = np.zeros([img_w, img_h, 3, 1])

# print("2")
_7_Transform = transforms.Compose([
    transforms.Resize(320),
    # transforms.RandomCrop(32,padding=4),
    transforms.RandomRotation(60,resample=False,expand=False,center=None),
    # transforms.RandomAffine(45,translate=[0.3,0.4],scale=[1,1],shear=None,resample=False,fillcolor=1)
])



def gen_txt(txt_path, img_dir):
    f = open(txt_path, 'w')
    for root, s_dirs, _ in os.walk(img_dir,topdown=True):  # 获取 train文件下各文件夹名称
        img_list = os.listdir(root)
        # print(len(img_list))
        # print(img_list)
        for i in range(len(img_list)):
           # 获取类别文件夹下所有png图片的路径
            if not img_list[i].endswith('png'):  # 若不是png文件，跳过
                continue
            label = img_list[i][0]
            img_path = os.path.join(root, img_list[i])
            line = img_path + ' ' + label + '\n'
            f.write(line)
    f.close()

def gen_transform(txt_dir,out_dir):
    with open(txt_dir,'r') as f:
        lines = f.readlines()
        # print(len(lines))
        for i in range(len(lines)) :
            img = cv2.imread(lines[i].rstrip().split()[0])
            img = cv2.resize(img,(img_h,img_w))
            img_name = lines[i].rstrip().split()[1]
            resize_path = os.path.join(_7_marker_trans,img_name)+'_resize.png'
            cv2.imwrite(resize_path,img)
            gen_txt(out_dir+'_resize.txt',out_dir)
    with open(out_dir+'_resize.txt','r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            img = cv2.imread(lines[i].rstrip().split()[0])
            for j in range(10):
                img = img.tranforms.RandomHorizontalFlip(p=0.5)
                img_name = lines[i].rstrip().split()[1]
                j_str = '-'+str(j)
                flap_path = os.path.join(_7_marker_trans,img_name)+j_str+'_flap.png'
                cv2.imwrite(flap_path,img)

class Mytrans(Dataset):
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
        _7_marker_trans_dir = '../../7_marker/trans/'
        fn, label = self.imgs[index]
        img = Image.open(fn)#.convert('RGB')     # 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1
        # img.show()
        if self.transform is not None:
            for i in range(10):
                img = self.transform(img)   # 在这里做transform，转为tensor等等
            # print(img)
                i_str = str(i)
                label_str = str(label)
                img.save(os.path.join(_7_marker_trans_dir,i_str)+'.png')

        return img, label

    def __len__(self):
        return len(self.imgs)



def gen_tranform_1(txt_dir_,out_dir,tranforms):
    imgs = Mytrans(txt_path=_7_marker_txt,transform=_7_Transform)

    img,label = imgs.__getitem__(5)
    # img.save(out_dir+'trans.png')


if __name__ == '__main__':
    # gen_txt(_7_marker_txt, _7_marker_img)

    # gen_transform(_7_marker_txt,_7_marker_trans)
    # gen_tranform_1(_7_marker_txt,_7_marker_trans,_7_Transform)
    imgs = Mytrans(txt_path=_7_marker_txt,transform=_7_Transform)
    img, label = imgs.__getitem__(5)
    # img.save('../../7_marker/trans/tans.png')