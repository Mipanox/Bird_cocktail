"""
 Utility functions
 - Mostly inherited from Stanford CS230 example code:
   https://github.com/cs230-stanford/cs230-code-examples/tree/master/pytorch/vision

"""

import json
import logging
import os
import shutil

import torch

class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
            
    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


class RunningAverage():
    """A simple class that maintains the running average of a quantity
    
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0
    
    def update(self, val):
        self.total += val
        self.steps += 1
    
    def __call__(self):
        return self.total/float(self.steps)
        
    
def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


##############################
# Modified WARP loss utility 
# - Reference: https://arxiv.org/pdf/1312.4894.pdf
import numpy as np
import math
import torch
import torch.nn as nn
from torch.autograd import Variable, Function
import random

class WARP(Function): 
    """
    Autograd function of WARP loss. Appropirate for multi-label
    - Reference: 
      https://medium.com/@gabrieltseng/intro-to-warp-loss-automatic-differentiation-and-pytorch-b6aa5083187a
    """
    @staticmethod
    def forward(ctx, input, target, max_num_trials = None):
        batch_size = target.size()[0]
        label_size = target.size()[1]

        ## rank weight 
        rank_weights = [1.0/1]
        for i in range(1, label_size):
            rank_weights.append(rank_weights[i-1] + (1.0/i+1))

        if max_num_trials is None: 
            max_num_trials = target.size()[1] - 1

        ##
        positive_indices = target.gt(0).float()
        negative_indices = target.eq(0).float()
        L = torch.zeros(input.size())

        for i in range(batch_size):
            for j in range(label_size):
                if target[i,j] == 1:
                    ## initialization
                    sample_score_margin = -1
                    num_trials = 0

                    while ((sample_score_margin < 0) and (num_trials < max_num_trials)):
                        ## sample a negative label, to only determine L (ranking weight)
                        neg_labels_idx = np.array([idx for idx, v in enumerate(target[i,:]) if v == 0])

                        if len(neg_labels_idx) > 0:                        
                            neg_idx = np.random.choice(neg_labels_idx, replace=False)
                            ## if model thinks neg ranks before pos...
                            sample_score_margin = input[i,neg_idx] - input[i,j]
                            num_trials += 1

                        else: # ignore cases where all labels are 1...
                            num_trials = 1
                            pass

                    ## how many trials determine the weight
                    r_j = int(np.floor(max_num_trials / num_trials))
                    L[i,j] = rank_weights[r_j]
        
        ## summing over all negatives and positives
        #-- since inputs are sigmoided, no need for clamp with min=0
        loss = torch.sum(L*(torch.sum(1 - positive_indices*input + \
                                          negative_indices*input, dim=1, keepdim=True)),dim=1)
        #ctx.save_for_backward(input, target)
        ctx.L = L
        ctx.positive_indices = positive_indices
        ctx.negative_indices = negative_indices
        
        return torch.sum(loss, dim=0)

    # This function has only a single output, so it gets only one gradient 
    @staticmethod
    def backward(ctx, grad_output):
        #input, target = ctx.saved_variables
        L = Variable(ctx.L, requires_grad = False)
        positive_indices = Variable(ctx.positive_indices, requires_grad = False) 
        negative_indices = Variable(ctx.negative_indices, requires_grad = False)

        pos_grad = torch.sum(L,dim=1,keepdim=True)*(-positive_indices)
        neg_grad = torch.sum(L,dim=1,keepdim=True)*negative_indices
        grad_input = grad_output*(pos_grad+neg_grad)

        return grad_input, None, None

#--- main class
class WARPLoss(nn.Module): 
    def __init__(self, max_num_trials = None): 
        super(WARPLoss, self).__init__()
        self.max_num_trials = max_num_trials
        
    def forward(self, input, target): 
        return WARP.apply(input.cpu(), target.cpu(), self.max_num_trials)


#############################
# Log-Sum-Exp-Pairwise Loss 
# - Reference: https://arxiv.org/pdf/1704.03135.pdf

def _to_one_hot(y, n_dims=None):
    """ 
    Take integer y (tensor or variable) with n dims and 
    convert it to 1-hot representation with n+1 dims
    """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(y.size()[0], -1)
    
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot

class LSEP(Function): 
    """
    Autograd function of LSEP loss. Appropirate for multi-label
    - Reference: Li+2017
      https://arxiv.org/pdf/1704.03135.pdf
    """
    
    @staticmethod
    def forward(ctx, input, target, max_num_trials = None):
        batch_size = target.size()[0]
        label_size = target.size()[1]

        ## rank weight 
        rank_weights = [1.0/1]
        for i in range(1, label_size):
            rank_weights.append(rank_weights[i-1] + (1.0/i+1))

        if max_num_trials is None: 
            max_num_trials = target.size()[1] - 1

        ##
        positive_indices = target.gt(0).float()
        negative_indices = target.eq(0).float()
        
        ## summing over all negatives and positives
        loss = 0.
        for i in range(input.size()[0]): # loop over examples
            pos = np.array([j for j,pos in enumerate(positive_indices[i]) if pos != 0])
            neg = np.array([j for j,neg in enumerate(negative_indices[i]) if neg != 0])
            
            for j,pj in enumerate(pos):
                for k,nj in enumerate(neg):
                    loss += np.exp(input[i,nj]-input[i,pj])
        
        loss = torch.from_numpy(np.array([np.log(1 + loss)])).float()
        
        ctx.save_for_backward(input, target)
        ctx.loss = loss
        ctx.positive_indices = positive_indices
        ctx.negative_indices = negative_indices
        
        return loss

    # This function has only a single output, so it gets only one gradient 
    @staticmethod
    def backward(ctx, grad_output):
        input, target = ctx.saved_variables
        loss = Variable(ctx.loss, requires_grad = False)
        positive_indices = ctx.positive_indices
        negative_indices = ctx.negative_indices

        fac  = -1 / loss
        grad_input = torch.zeros(input.size())
        
        ## make one-hot vectors
        one_hot_pos, one_hot_neg = [],[]
        
        for i in range(grad_input.size()[0]): # loop over examples
            pos_ind = np.array([j for j,pos in enumerate(positive_indices[i]) if pos != 0])
            neg_ind = np.array([j for j,neg in enumerate(negative_indices[i]) if neg != 0])
            
            one_hot_pos.append(_to_one_hot(torch.from_numpy(pos_ind),input.size()[1]))
            one_hot_neg.append(_to_one_hot(torch.from_numpy(neg_ind),input.size()[1]))
            
        ## grad
        for i in range(grad_input.size()[0]):
            for dum_j,phot in enumerate(one_hot_pos[i]):
                for dum_k,nhot in enumerate(one_hot_neg[i]):
                    grad_input[i] += (phot-nhot)*torch.exp(-input[i].data*(phot-nhot))
        ## 
        grad_input = Variable(grad_input) * (grad_output * fac)

        return grad_input, None, None
    
#--- main class
class LSEPLoss(nn.Module): 
    def __init__(self): 
        super(LSEPLoss, self).__init__()
        
    def forward(self, input, target): 
        return LSEP.apply(input.cpu(), target.cpu())



####################
# confusion matrix #
import torch.nn.functional as F
import sklearn.metrics as met

def confusion_matrix(outputs, labels):
    nlabels = outputs.shape[1]
    
    outputs = np.argmax(F.softmax(outputs, dim=1).data.cpu().numpy(),axis=1)
    labels  = labels.data.cpu().numpy()
    
    cm = met.confusion_matrix(labels, outputs, np.arange(nlabels))
    cm = cm.astype('float')*10 / cm.sum(axis=1)[:, np.newaxis]
    cm = np.nan_to_num(cm, copy=True)
    cm = cm.astype('int')
    
    return cm