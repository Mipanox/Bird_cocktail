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
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.datasets import *
from torchvision.datasets.folder import *
import torchvision.transforms as transforms


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
                 aug_factor=1, mixing=True, num_mix=5):
        """
        Options:
        - aug_factor : integer
          If number of original dataset is N, generates N * aug_factor datasets
          Must be 1 if mixing is False
          
        - mixing : bool
        - num_mix : integer
          If mixing = True, synthesize num_mix randomized bird sounds
        """
        
        classes, class_to_idx = find_classes(root)
        imgs       = make_dataset( root, class_to_idx)
        noise_imgs = make_noiseset(noise_root)
        
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.noise_imgs = noise_imgs
        self.aug_factor = aug_factor
        self.mixing  = mixing
        self.num_mix = num_mix
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = grey_loader
        
        ##
        assert isinstance(self.aug_factor, int), "aug_factor must be an integer"
        assert isinstance(self.num_mix, int), "num_mix must be an integer"
        if not self.mixing and (self.aug_factor > 1):            
            raise ValueError('No mixing is mutually exclusive with aug_factor > 1')
            
    def __len__(self):
        return len(self.imgs) * self.aug_factor
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is multi-hot vector of the target class.
        """
        tot_len = self.__len__()
        
        ## initialize superposed img array as the noise
        np.random.seed(index) # use index as random seed to ensure reproducibility

        path  = self.noise_imgs[np.random.randint(0,tot_len/self.aug_factor) % len(self.noise_imgs)]
        noise = self.loader(path)
        if self.transform is not None:          
            noise = self.transform(noise)
        
        img_res_ar = np.array(noise) * 0.1  # not too strong        
        tar_res_lt = []
        
        ##
        if self.mixing: # if synthesizing spectrograms randomly            
            num_item = np.random.randint(1,self.num_mix+1) # min 1 bird, max 5 birds
            strength = np.random.rand(num_item) # weights of superposed species
            #                                     (note this is in intensity)
            for num in range(num_item):
                idx = np.random.randint(0,tot_len/self.aug_factor)
            
                ## load image
                path, target = self.imgs[idx]
                img = self.loader(path)            
                if self.transform is not None:
                    img = self.transform(img)
                
                ## array
                img_res_ar += np.array(img)*strength[num]
                tar_res_lt.append(target)
        
            ## rescaling to [0,255]
            img_res_ar -=  img_res_ar.min()
            if img_res_ar.max() == 0:
                img_res_ar = np.zeros(img_res_ar.shape) + 1e-5
            else:
                img_res_ar /= (img_res_ar.max() / 255.)

            ## convert to multi-hot vector
            tar_res_lt = code_to_vec(sorted(set(tar_res_lt)),self.classes)

        else:
            path, target = self.imgs[index]
            img = self.loader(path)
            if self.transform is not None:
                img = self.transform(img)
                
            img_res_ar += np.array(img)

            ## as class values [0, num_classes - 1]
            tar_res_lt = target
            
        img = transforms.ToTensor().__call__(img_res_ar.transpose(1,2,0)) # back to [0,1]

        return img, tar_res_lt

#################
## data loader ##

def fetch_dataloader(types, data_dir, params, **kwargs):
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

    # transformations, tunable
    ## define a training image loader that specifies transforms on images.
    train_transformer = transforms.Compose([
        transforms.RandomCrop((128, params.width)), # keep freq axis (0) fixed to 128
        transforms.ToTensor()])            # transform it into a torch tensor

    ## loader for evaluation (possibly different from train_transformer...)
    eval_transformer = transforms.Compose([
        transforms.RandomCrop((128, params.width)), # keep freq axis (0) fixed to 128
        transforms.ToTensor()])            # transform it into a torch tensor

    ## loader for test (possibly different from train_transformer...)
    test_transformer = transforms.Compose([
        transforms.RandomCrop((128, params.width)), # keep freq axis (0) fixed to 128
        transforms.ToTensor()])            # transform it into a torch tensor


    for split in ['train', 'val', 'test']:
        if split in types:
            path       = os.path.join(data_dir, "{}".format(split))
            noise_path = os.path.join(data_dir, "{}".format('noise'))

            # use the train_transformer if training data, else use eval_transformer without random flip
            if split == 'train':
                dl = DataLoader(BirdFolder(path, noise_path, train_transformer, 
                                aug_factor=params.aug_factor, **kwargs),
                                batch_size=params.batch_size, shuffle=True,
                                num_workers=params.num_workers,
                                pin_memory=params.cuda)
            elif split == 'val':
                dl = DataLoader(BirdFolder(path, noise_path, eval_transformer, 
                                aug_factor=params.aug_factor, **kwargs), 
                                batch_size=params.batch_size, shuffle=False,
                                num_workers=params.num_workers,
                                pin_memory=params.cuda)
            else: # test
                dl = DataLoader(BirdFolder(path, noise_path, test_transformer, 
                                aug_factor=params.aug_factor, **kwargs), 
                                batch_size=params.batch_size, shuffle=False,
                                num_workers=params.num_workers,
                                pin_memory=params.cuda)

            dataloaders[split] = dl

    return dataloaders