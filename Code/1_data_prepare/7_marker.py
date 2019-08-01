# coding:utf-8
import torch
import torchvision.transforms as transforms
import numpy as np
import os
import sys
import cv2

_7_marker_img = '/home/sun/Documents/7_marker/'
_7_marker_txt = '/home/sun/Documents/7_marker/7_marker.txt'
_7_marker_trans = '/home/sun/Documents/7_marker/'

img_h, img_w = 320, 320
imgs = np.zeros([img_w, img_h, 3, 1])

# print("2")

def gen_txt(txt_path, img_dir):
    f = open(txt_path, 'w')
    for root, s_dirs, _ in os.walk(img_dir,topdown=True):  # 获取 train文件下各文件夹名称
        img_list = os.listdir(root)
        for i in range(len(_)):
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
        for i in range(len(lines)) :
            img = cv2.imread(lines[i].rstrip().split()[0])



if __name__ == '__main__':
    gen_txt(_7_marker_txt, _7_marker_img)



# trainTransform = transforms.Compose([
#     transforms.Resize(32),
#     transforms.RandomCrop(32,padding=4),
#     transforms.ToTensor(),
#     normalTransform
# ])
# validTransform = transforms.Compose([
#     transforms.ToTensor(),
#     normalTransform
# ])