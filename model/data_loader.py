"""
 Data loader class definition. 
 - Largely inherited from Stanford CS230 example code:
   https://github.com/cs230-stanford/cs230-code-examples/tree/master/pytorch/vision
  
 - Also borrowed from http://pytorch.org/tutorials/advanced/neural_style_tutorial.html
   and http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""


import random
import os
import numpy as np

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.datasets import *
import torchvision.transforms as transforms


## define a training image loader that specifies transforms on images.
train_transformer = transforms.Compose([
    transforms.RandomCrop((128, 192)), # keep freq axis (0) fixed to 128
    transforms.ToTensor()])            # transform it into a torch tensor

## loader for evaluation (possibly different from train_transformer...)
eval_transformer = transforms.Compose([
    transforms.RandomCrop((128, 192)), # keep freq axis (0) fixed to 128
    transforms.ToTensor()])            # transform it into a torch tensor


######################
## Helper functions ##

def grey_loader(path):
    """
    Images are grey scale
    """
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')
    
def make_noiseset(dir):
    """
    Load noises from files
    """
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = path
                    images.append(item)
    
    return images   

def code_to_vec(code,classes):
    """
    Convert list of target class labels to multi-hot vector (tensor)
    """
    def char_to_vec(c):
        y = np.zeros((len(classes),))
        for cc in c:
            y[cc] = 1.0
        return y
    c = torch.from_numpy(char_to_vec(code))
    return c



##############################
## Custom ImageFolder class ## 

class BirdFolder(ImageFolder):
    """
    A dataloader object inherited from PyTorch's ImageFolder:
     http://pytorch.org/docs/master/torchvision/datasets.html#imagefolder
    """

    def __init__(self, root, noise_root, transform=None, target_transform=None,
                 loader=default_loader):
        
        classes, class_to_idx = find_classes(root)
        imgs       = make_dataset( root, class_to_idx)
        noise_imgs = make_noiseset(noise_root)
        
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.noise_imgs = noise_imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = grey_loader
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        tot_len = self.__len__()
        num_item = np.random.randint(1,5+1) # min 1 bird, max 5 birds
        strength = np.random.rand(num_item) # weights of superposed species
        #                                     (note this is in intensity)
        
        ## initialize superposed img array as the noise
        path  = self.noise_imgs[index]
        noise = self.loader(path)
        if self.transform is not None:            
            noise = self.transform(noise)
        
        img_res_ar = np.array(noise) * 0.1  # not too strong        
        tar_res_lt = []
        for num in range(num_item):
            idx = np.random.randint(0,tot_len)
            
            ## load image
            path, target = self.imgs[idx]
            img = self.loader(path)            
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            
            ## array
            img_res_ar += np.array(img)*strength[num]
            tar_res_lt.append(target)
        
        ## rescaling to [0,255]
        img_res_ar -=  img_res_ar.min()
        img_res_ar /= (img_res_ar.max() / 255)        
        
        #img_res_ar = np.transpose(img_res_ar,(2,0,1))
        img = transforms.ToTensor().__call__(img_res_ar.transpose(1,2,0)) # back to [0,1]
        
        tar_res_lt = sorted(set(tar_res_lt))
        
        return img, tar_res_lt

#################
## data loader ##

def fetch_dataloader(types, data_dir, params=None):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}

    for split in ['train', 'val', 'test']:
        if split in types:
            path       = os.path.join(data_dir, "{}".format(split))
            noise_path = os.path.join(data_dir, "{}".format('noise'))

            # use the train_transformer if training data, else use eval_transformer without random flip
            if split == 'train':
                dl = DataLoader(BirdFolder(path, noise_path, train_transformer), shuffle=True)
            else:
                dl = DataLoader(BirdFolder(path, noise_path, eval_transformer ), shuffle=False)

            dataloaders[split] = dl

    return dataloaders        