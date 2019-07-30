import os

train_txt_path = '../../Data/train.txt'
train_dir = '../../Data/train/'

valid_txt_path = '../../Data/valid.txt'
valid_dir = '../../Data/valid/'

def generate_txt(text_path,img_dir):
    fo_ = open(text_path,'w')
    for root,s_dirs,_ in os.walk(img_dir,topdown=True):
        for subdirs in s_dirs:
            i_dir = os.path.join(root,subdirs)
            print (i_dir)
            img_list = os.listdir(i_dir)
            print(img_list)
            for i in range(len(img_list)):
                if not img_list[i].endswith('.png'):
                    continue
                label = img_list[i].split('_')[0]
                img_path = os.path.join(i_dir,img_list[i])
                fo_.write(img_path+' '+label+'\n')
    fo_.close()

if __name__ == '__main__':
    generate_txt(train_txt_path,train_dir)