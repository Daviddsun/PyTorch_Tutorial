import os
import glob
import shutil
import random

dataset_dir = '../../Data/cifar-10-png/raw_test/'
train_dir = '../../Data/train/'
valid_dir = '../../Data/valid/'
test_dir = '../../Data/test/'

train_per = 0.8
valid_per = 0.1
test_per = 0.1

def makedir(dir_):
    if not os.path.exists(dir_):
        os.makedirs(dir_)

if __name__ == '__main__':
    for root, dirs, files in os.walk(dataset_dir):
        for sDir in dirs:
            img_list = glob.glob(os.path.join(root,sDir)+"/*.png")
            # print (root)
            # print (sDir)
            # print (img_list)
            random.seed(95)
            random.shuffle(img_list)
            img_num = len(img_list)

            train_point = int( img_num * train_per)
            valid_point = int( train_point + img_num * valid_per)

            for i in range(img_num):
                if i < train_point :
                    out_dir = str(os.path.join(train_dir,sDir)) + "/"
                elif  i < valid_point:
                    out_dir = valid_dir + sDir +"/"
                else :
                    out_dir = test_dir + sDir + "/"
                makedir(out_dir)
                out_path = out_dir + os.path.split(img_list[i])[-1]
                # print(os.path.split(img_list[i])[-1])
                # print(out_path)
                shutil.copy(img_list[i],out_path)
                # print(out_path)
            print('class:{},train:{},valid:{},test:{}'.format(sDir,train_point,valid_point-train_point,img_num-valid_point))





