"""
Utility functions for demonstration or illustration. Mostly copied from individual python scripts
"""
from __future__ import print_function
import numpy as np
import cv2
import librosa
import scipy.ndimage as ndimage
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.datasets import *
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def getSpec(sig, rate, power=2, **kwargs):
    """
    Get spectrogram from signal split
    
    Options:
    - power : integer, 1 or 2
      Specify the power of spectrogram. Defaults to intensity
      For power = 1, return amplitude spectrogram
    """
    
    ## melspetrogram
    magspec = librosa.feature.melspectrogram(y=sig,sr=rate,power=power,**kwargs)
    
    return magspec


def getMultiSpec(path, seconds=3., overlap=2.5, minlen=3., **kwargs):
    """
    Split signal into chunks with overlap of certain seconds and tunable minimum length
    
    Options:
    - seconds : np.float
      Length of output chunk (segment) of audio. Defaults to 3 seconds
    
    - overlap : np.float
      Length of overlapping window of segmentation. Defaults to 2.5 seconds
      
    - minlen : np.float
      Minimal length of output spectrogram in seconds. Defaults to 3 seconds.
    """
    #--
    ## open audio file
    sig, rate = librosa.load(path,sr=None) # natural sample rate
    
    ## adjust to different sample rates
    if rate != 44100:
        sig  = librosa.resample(sig, rate, 44100)

    #--
    ## split signal with overlap
    sig_splits = []
    for i in xrange(0, len(sig), int((seconds - overlap) * rate)):
        split = sig[i:i + int(seconds*rate)]
        if len(split) >= minlen * rate:
            sig_splits.append(split)

    ## if signal too short for segmentation, append it anyway
    if len(sig_splits) == 0:
        sig_splits.append(sig)
    
    #--
    ## calculate spectrogram for each split
    for sig in sig_splits:
        magspec = getSpec(sig, rate, **kwargs)

        yield magspec


def hasBird(spec, raw_thresh=10., threshold=30.):
    """
    Decide if given spectrum shows bird sounds or noise only
    
    Options:
    - raw_thresh : float
      Threshold for the absolute value of signal before fancy processing
      Defaults to 10
    
    - threshold : float
      Threshold value for separating real signal / noise. Defaults to 30.
      Changeable via command-line option (see above)
    """
    ## working copy
    img = spec.copy()
    
    ### STEP 0: get rid of highest/lowest freq bins
    #-- total of 128 bins, to integer vals due to JPEG format
    img = img[20:100,:].astype(int).astype(np.float32) 
    
    bird, rthresh = False, 0. # if too weak, treat as no signal no matter what
    if img.max() > raw_thresh: # absolute value of maximum signal
        ### STEP 1: normalize to [0,1] for processing
        img -= img.min()
        img /= img.max()
    
        ### STEP 2: Median blur
        img = cv2.medianBlur(img,5)

        ### STEP 3: Median threshold
        col_median = np.median(img, axis=0, keepdims=True)
        row_median = np.median(img, axis=1, keepdims=True)

        img[img < row_median * 3] = 0.
        img[img < col_median * 4] = 0.
        img[img > 0.] = 1.

        ### STEP 4: Morph Closing
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((5,5), np.float32))
    
        ### STEP 5: Count columns and rows with signal
        #-- (Note: We only use rows with signal as threshold -- time axis for fixed freqs)
    
        ##### row has signal?
        row_max = np.max(img, axis=1)
        row_max = ndimage.morphology.binary_dilation(row_max, iterations=2).astype(row_max.dtype)
        rthresh = row_max.sum()
    
        ### STEP 6: Apply threshold
        if rthresh >= threshold:
            bird = True
    
    return bird, rthresh, img

###################################################################################
#--- some ImageFolder source codes and data_loader definitions
#     https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

def is_image_file(filename):
    """Checks if a file is an image.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
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
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images

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

train_transformer = transforms.Compose([
    transforms.RandomCrop((128, 192)), # keep freq axis (0) fixed to 128
    transforms.ToTensor()])            # transform it into a torch tensor
    
class BirdFolder(ImageFolder):
    """
    A dataloader object inherited from PyTorch's ImageFolder:
     http://pytorch.org/docs/master/torchvision/datasets.html#imagefolder
    """

    def __init__(self, root, noise_root, aug_factor=1, mixing=True, num_mix=5,
                 transform=None, target_transform=None):
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
        print('Total number of data: {}'.format(tot_len))
        ## initialize superposed img array as the noise
        path  = self.noise_imgs[np.random.randint(0,tot_len/self.aug_factor)]
        noise = self.loader(path)
        if self.transform is not None:          
            noise = self.transform(noise)
        print(' chosen noise to be added to signal: {}'.format(path.split('/')[-1]))

        img_res_ar = np.array(noise) * 0.1  # not too strong        
        tar_res_lt = []
        
        ##
        if self.mixing: # if synthesizing spectrograms randomly
            np.random.seed(index)
            num_item = np.random.randint(1,self.num_mix+1) # min 1 bird, max 5 birds
            strength = np.random.rand(num_item) # weights of superposed species
            #                                     (note this is in intensity)
            print(' {0} spectrograms are drawn with strengths {1}'.format(num_item,strength))
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
            print(' and they are from species: {}'.format(tar_res_lt))
            ## rescaling to [0,255]
            img_res_ar -=  img_res_ar.min()
            img_res_ar /= (img_res_ar.max() / 255)        
            
        else:
            path, target = self.imgs[index]
            img = self.loader(path)
            if self.transform is not None:
                img = self.transform(img)
                
            img_res_ar += np.array(img)
            tar_res_lt.append(target)
            
        img = transforms.ToTensor().__call__(img_res_ar.transpose(1,2,0)) # back to [0,1]
        tar_res_lt = code_to_vec(sorted(set(tar_res_lt)),self.classes)
        
        return img, tar_res_lt

from mpl_toolkits import axes_grid1
def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot"""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)