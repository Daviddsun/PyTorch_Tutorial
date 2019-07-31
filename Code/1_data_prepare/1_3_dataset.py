from PIL import Image
from torch.utils.data import Dataset

class get_dataset(Dataset):
    def __init__(self,txt_path,transform =None,target_transform=None):
        fh = open(txt_path,'r')
        imgs = []
        for line in fh:
            line = line.restrip()
            words = line.split()




