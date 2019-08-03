# coding:utf-8
# import torch
# from torch.utils.data import Dataset
# import torchvision.transforms as transforms
import numpy as np
import os
import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
# from utils.utils import MyDataset
from PIL import Image
import matplotlib.pyplot as plt
import copy

_7_marker_img = '../../7_marker/'
_7_marker_txt = '../../7_marker/7_marker.txt'
_7_marker_trans = '../../7_marker/trans/'
_resize_dir = '../../7_marker/trans/resize/'
_resize_txt = '../../7_marker/trans/resize/_resize.txt'
_backgroud_dir = '../../7_marker/trans/backgroud/'
_backgroud_txt = '../../7_marker/trans/backgroud/_background.txt'

_affine_dir = '../../7_marker/trans/affine/'

_attach_dir = '../../7_marker/trans/attach/'
_label_txt= '../../7_marker/trans/attach/_lable.txt'

img_h, img_w = 320, 320
img_zero_x,img_zero_y = 500,500
img_zeros = np.zeros([img_zero_x, img_zero_y, 3])
# cv2.imshow('1',imgs)
# cv2.waitKey()
# print("2")
# _7_Transform = transforms.Compose([
#     transforms.Resize(320),
#     # transforms.RandomCrop(32,padding=4),
#     transforms.RandomRotation(60,resample=False,expand=False,center=None),
#     transforms.RandomAffine(45,translate=None,scale=None,shear=None,resample=False,fillcolor=1)
#
# ])


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

def gen_resize(txt_dir,resize_path_):
    with open(txt_dir,'r') as f:
        lines = f.readlines()
        # print(len(lines))
        for i in range(len(lines)) :
            img = cv2.imread(lines[i].rstrip().split()[0])
            img = cv2.resize(img,(img_h,img_w))
            img_name = lines[i].rstrip().split()[1]
            resize_path = os.path.join(resize_path_,img_name)+'_resize.png'
            cv2.imwrite(resize_path,img)
            gen_txt(resize_path_+'_resize.txt',resize_path_)

def gen_affine(resized_dir,back_groud_dir,_attach_dir_,num_gen):
    with open(resized_dir+'_resize.txt','r') as f:
        lines_mk = f.readlines()
        x_mk_zero_1,x_mk_zero_2,y_mk_zero_1,y_mk_zero_2 =(img_zero_x-img_w)/2,(img_zero_x+img_w)/2,(img_zero_y-img_h)/2,(img_zero_y+img_h)/2

        for i_mk in range(len(lines_mk)):
            img_mk_small = cv2.imread(lines_mk[i_mk].rstrip().split()[0],-1)
            # print (img_mk_small.shape)
            img_mk = copy.deepcopy(np.array(img_zeros))
            img_mk[x_mk_zero_1:x_mk_zero_2,y_mk_zero_1:y_mk_zero_2 ,:][img_mk_small > 10]=img_mk_small[img_mk_small > 10]
            # print(img_mk.shape)
            rows,cols,channels= img_mk.shape[:3]
            with open(back_groud_dir+'_background.txt','r') as b_g :
                lines_bg = b_g.readlines()
                for i_b in range(len(lines_bg)):
                    img_bg = cv2.imread(lines_bg[i_b].rstrip().split()[0])
                    rows_mk,cols_mk = img_bg.shape[:2]
                    # print(rows_mk,cols_mk)
                    window_mngr= plt.figure(figsize=(300,200))
                    window_mngr.canvas.manager.window.move(180,110)
                    # plt.figure(figsize=(300,300))
                    for j in range(num_gen):
                        a = np.random.uniform(0.1,0.3,2)
                        b = np.random.uniform(0.7,0.9,2)
                        c = np.random.uniform(0.1,0.2,2)
                        [n1,n2]=a
                        [n3,n6]=b
                        [n5,n4]=c
                        # print(n1)
                        pts_aff_1_ = np.float32([[0,0],[cols-1,0],[0,rows-1]])
                        pts_aff_2_ = np.float32([[cols*n1,rows*n2],[cols*n3,rows*n4],[cols * n5, rows * n6]])
                        pts_per_1 = np.float32([[56, 65], [238, 52], [28, 237], [239, 240]])
                        pts_per_2 = np.float32([[0, 0], [200, 0], [0, 200], [200, 200]])

                        M_aff = cv2.getAffineTransform(pts_aff_1_,pts_aff_2_)
                        M_per = cv2.getPerspectiveTransform(pts_per_1, pts_per_2)

                        corner_0 = np.dot(M_aff,(x_mk_zero_1,y_mk_zero_1,1))     # M .* [x,y,1].T
                        corner_2 = np.dot(M_aff,(x_mk_zero_2,y_mk_zero_2,1))
                        corner_1 = np.dot(M_aff,(x_mk_zero_1,y_mk_zero_2,1))
                        corner_3 = np.dot(M_aff,(x_mk_zero_2,y_mk_zero_1,1))

                        corner_list = [[corner_0[0],corner_0[1]],[corner_1[0],corner_1[1]],[corner_2[0],corner_2[1]],[corner_3[0],corner_3[1]]]

                        # print(corner_1[2])
                        dst_aff = cv2.warpAffine(img_mk,M_aff,(cols,rows))
                        dst_per = cv2.warpPerspective(dst_aff,M_per,(cols,rows))
                        # plt.ion()
                        plt.subplot(231)
                        plt.imshow(dst_aff)
                        plt.subplot(232)
                        plt.imshow(dst_per)
                        # plt.show()




                        # img_mk_mask = np.zeros(img_mk.shape,img_mk.dtype)
                        # # print(img_mk.dtype)
                        # img_bg_mask = np.zeros(img_bg.shape,img_bg.dtype)
                        # poly = np.array(corner_list,np.int32)
                        # # print(img_bg.dtype)
                        # # print(poly)
                        # cv2.fillPoly(img_mk_mask, [poly], (255, 255, 255))
                        #
                        # print(rows_mk*0.1+img_h*0.5,rows_mk*0.9-img_h*0.5,cols_mk*0.1+img_w*0.5,cols_mk*0.9-img_w*0.5)
                        [center_x,center_y] = [np.random.randint(rows_mk*0.1+img_zero_x*0.5,rows_mk*0.9-img_zero_x*0.5) ,np.random.randint(cols_mk*0.1+img_zero_y*0.5,cols_mk*0.9-img_zero_y*0.5)]

                        # center = (center_x,center_y)
                        # center_list=[center,center,center,center]

                        # #----- 方法0、 seamlessClone -----#
                        # attach = cv2.seamlessClone(dst_aff, img_bg,img_mk_mask, center, cv2.MIXED_CLONE)

                        #----- 方法一、 * 运算 ------#
                        # outPutImg = img_bg.copy()
                        # bg_roi = outPutImg[center[1]:center[1] + img_h,center[0]:center[0] + img_w] # 背景 范围
                        #
                        # bg_poly = poly + np.array(center_list)  # marker 变换后抠图区域 在 背景的 占位区域
                        # cv2.fillPoly(outPutImg,[bg_poly],(0,0,0))  # 背景上占位区域->黑
                        #
                        # img_mask_array = img_mk_mask*img_mk          # marker 变换抠图 0* x= 0
                        #
                        # output_roi_mk = bg_roi+img_mask_array       # 去除marker 黑边
                        # outPutImg[center[1]:center[1] + 320,center[0]:center[0] + 320]=output_roi_mk

                        #-------方法二、 位运算 ------#
                        # bg_mask = img_bg_mask[np.min(bg_poly[:,1]):np.max(bg_poly[:,1]),np.min(bg_poly[:,0]):np.max(bg_poly[:,0])]
                        # print(poly[:,1])
                        # print(poly)
                        # bg_roi.astype('int32')
                        # bg = cv2.bitwise_and(bg_roi,bg_roi,mask = bg_mask)
                        # marker_cut = img_mk[np.min(poly[:,1]):np.max(poly[:,1]),np.min(poly[:,0]):np.max(poly[:,0])]

                        #--------- 方法三、 np.array()[] []
                        imgcopy = copy.deepcopy(np.array(img_bg))
                        # print([dst_per>10])
                        x1,x2 = center_x-img_zero_x/2,center_x-img_zero_x/2+img_zero_x
                        y1,y2 = center_y-img_zero_y/2,center_y-img_zero_y/2+img_zero_y
                        # xxx=corner_list+[[x1,y1]]
                        # print(corner_list)
                        # print(corner_list+[[x1,y1]])
                        # print(center_x,center_y)
                        # print(x1,x2,y1,y2)
                        imgcopy[x1:x2,y1:y2,:][dst_per>10]=dst_per[dst_per>10]

                        img_name = lines_mk[i_mk].rstrip().split()[1]
                        # aff_path = os.path.join(affine_out_dir,img_name)+'_'+str(j)+'_affine.png'
                        out_path = os.path.join(_attach_dir_, img_name) + str(j) + '_.png'

                        # cv2.imwrite(aff_path,dst_aff)
                        cv2.imwrite(out_path,imgcopy)
                        # line = out_path+' '+

                        plt.subplot(233)
                        # plt.imshow(attach)
                        plt.subplot(234)
                        # plt.imshow(outPutImg)
                        plt.subplot(235)
                        # plt.imshow(imgcopy_test)
                        plt.subplot(236)
                        plt.imshow(imgcopy)
                        plt.show()
                        # cv2.imshow('1',mk1)
                        # cv2.imshow('1',imgcopy)
                        # cv2.waitKey()

            # plt.ioff()
            plt.clf()
            plt.close()

#--------  use torch transform ---------#
# class Mytrans(Dataset):
#     def __init__(self, txt_path, transform = None, target_transform = None):
#         fh = open(txt_path, 'r')
#         imgs = []
#         for line in fh:
#             line = line.rstrip()
#             words = line.split()
#             imgs.append((words[0], int(words[1])))
#
#         self.imgs = imgs        # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
#         self.transform = transform
#         self.target_transform = target_transform
#
#     def __getitem__(self, index):
#         _7_marker_trans_dir = '../../7_marker/trans/'
#         fn, label = self.imgs[index]
#         img = Image.open(fn)#.convert('RGB')     # 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1
#         # img.show()
#         if self.transform is not None:
#             for i in range(10):
#                 img = self.transform(img)   # 在这里做transform，转为tensor等等
#             # print(img)
#                 i_str = str(i)
#                 label_str = str(label)
#                 img.save(os.path.join(_7_marker_trans_dir,i_str)+'.png')
#
#         return img, label
#
#     def __len__(self):
#         return len(self.imgs)
#
#
#
# def gen_tranform_1(num,txt_path_,out_dir,transform_):
#     for j in range(7):
#         for i in range(num):
#             imgs = Mytrans(txt_path=txt_path_,transform=transform_)
#             img,label = imgs.__getitem__(j)
#             img.save(out_dir+str(label)+'_'+str(i)+'_'+'trans.png')
#
#--------  end of use torch transform ---------#


if __name__ == '__main__':
    # gen_txt(_7_marker_txt, _7_marker_img)

    gen_resize(_7_marker_txt,_resize_dir)
    gen_txt(_backgroud_txt,_backgroud_dir)
    # gen_tranform_1(1, _7_marker_txt,_7_marker_trans,_7_Transform)
    gen_affine(_resize_dir,_backgroud_dir,_attach_dir,1)