from scipy.misc import imsave
import os
import numpy as np
import pickle

data_dir = os.path.join("..", "..", "Data", "cifar-10-batches-py")
train_o_dir = os.path.join("..", "..", "Data", "cifar-10-png", "raw_train")
test_o_dir = os.path.join("..", "..", "Data", "cifar-10-png", "raw_test")

print(data_dir)

def mydir(my_dir):
    if not os.path.isdir(my_dir):
        os.makedirs(my_dir)
def unpickle(file):
    with open (file,'rb') as fo_:
        dict_ = pickle.load(fo_,encoding='bytes')
    return dict_
test_data_path = os.path.join(data_dir,"test_batch")
test_data = unpickle(test_data_path)

for i in range (0,10000):
    img = np.reshape(test_data[b'data'][i],(3,32,32))
    img = img.transpose(1,2,0)

    label_num = str(test_data[b'labels'][i])
    out_dir = os.path.join(test_o_dir+'_0',label_num)

    mydir(out_dir)

    img_name = label_num+"_"+str(i)+".png"
    img_path = out_dir+"/"+img_name
    imsave(img_path,img)
