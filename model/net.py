"""
 Defines the neural network, losss function and metrics
 - Largely inherited from Stanford CS230 example code:
   https://github.com/cs230-stanford/cs230-code-examples/tree/master/pytorch/vision
 - Partly borrowed from andreaazzini's repository:
   https://github.com/andreaazzini/multidensenet
 - Partly borrowed from official PyTorch model source code:
   https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1
 - Partly borrowed from Cadene's github repository:
   https://github.com/Cadene/tensorflow-model-zoo.torch

"""

import numpy as np
import sklearn.metrics as met
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.models as models

import sys
import math

# DenseNet

######################
## Helper functions ##
class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out

class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out

class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out


#############################
## Main Network Definition ##
class DenseNetBase(nn.Module):
    """
    Documentation for reference: http://pytorch.org/docs/master/nn.html
    """

    def __init__(self, params, num_classes):
        """
        Args:
            params: (Params) contains growthRate, depth, reduction, bottleneck
        """
        super(DenseNetBase, self).__init__()

        growthRate = params.growthRate
        depth      = params.depth
        reduction  = params.reduction
        nClasses   = num_classes
        bottleneck = True #params.bottleneck

        nDenseBlocks = (depth-4) // 3
        if bottleneck:
            nDenseBlocks //= 2

        nChannels = 2*growthRate
        self.conv1 = nn.Conv2d(1, nChannels, kernel_size=3, padding=1, bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans1 = Transition(nChannels, nOutChannels)
        
        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans2 = Transition(nChannels, nOutChannels)
        
        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate

        self.bn1 = nn.BatchNorm2d(nChannels)
        ##                             input channel determined by dimension of imgs
        self.fc = nn.Linear(nChannels*(128*params.width)/32**2, nClasses)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        """ Construct dense layers """

        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate

        return nn.Sequential(*layers)

    def forward(self, s):
        """
        Args:
            s: (Variable) contains a batch of images, of dimension batch_size x 1 x 128 x params.width

        Returns:
            out: (Variable) dimension batch_size x num_classes
        """
        out = self.conv1(s)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))
        out = out.view(out.size()[0], -1)
        return self.fc(out)

# Inception-v4

######################
## Helper functions ##
class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False) # verify bias false
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001, momentum=0, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Mixed_3a(nn.Module):

    def __init__(self):
        super(Mixed_3a, self).__init__()
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.conv = BasicConv2d(64, 96, kernel_size=3, stride=2)

    def forward(self, x):
        x0 = self.maxpool(x)
        x1 = self.conv(x)
        out = torch.cat((x0, x1), 1)
        return out

class Mixed_4a(nn.Module):

    def __init__(self):
        super(Mixed_4a, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 64, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(64, 64, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(64, 96, kernel_size=(3,3), stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        return out

class Mixed_5a(nn.Module):

    def __init__(self):
        super(Mixed_5a, self).__init__()
        self.conv = BasicConv2d(192, 192, kernel_size=3, stride=2)
        self.maxpool = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.conv(x)
        x1 = self.maxpool(x)
        out = torch.cat((x0, x1), 1)
        return out

class Inception_A(nn.Module):

    def __init__(self):
        super(Inception_A, self).__init__()
        self.branch0 = BasicConv2d(384, 96, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(384, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(384, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1),
            BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(384, 96, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out

class Reduction_A(nn.Module):

    def __init__(self):
        super(Reduction_A, self).__init__()
        self.branch0 = BasicConv2d(384, 384, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv2d(384, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 224, kernel_size=3, stride=1, padding=1),
            BasicConv2d(224, 256, kernel_size=3, stride=2)
        )
        
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out

class Inception_B(nn.Module):

    def __init__(self):
        super(Inception_B, self).__init__()
        self.branch0 = BasicConv2d(1024, 384, kernel_size=1, stride=1)
        
        self.branch1 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 224, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(224, 256, kernel_size=(7,1), stride=1, padding=(3,0))
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(192, 224, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(224, 224, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(224, 256, kernel_size=(1,7), stride=1, padding=(0,3))
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(1024, 128, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out

class Reduction_B(nn.Module):

    def __init__(self):
        super(Reduction_B, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=3, stride=2)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(1024, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(256, 320, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(320, 320, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out

class Inception_C(nn.Module):

    def __init__(self):
        super(Inception_C, self).__init__()

        self.branch0 = BasicConv2d(1536, 256, kernel_size=1, stride=1)
        
        self.branch1_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch1_1a = BasicConv2d(384, 256, kernel_size=(1,3), stride=1, padding=(0,1))
        self.branch1_1b = BasicConv2d(384, 256, kernel_size=(3,1), stride=1, padding=(1,0))
        
        self.branch2_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch2_1 = BasicConv2d(384, 448, kernel_size=(3,1), stride=1, padding=(1,0))
        self.branch2_2 = BasicConv2d(448, 512, kernel_size=(1,3), stride=1, padding=(0,1))
        self.branch2_3a = BasicConv2d(512, 256, kernel_size=(1,3), stride=1, padding=(0,1))
        self.branch2_3b = BasicConv2d(512, 256, kernel_size=(3,1), stride=1, padding=(1,0))
        
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(1536, 256, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        
        x1_0 = self.branch1_0(x)
        x1_1a = self.branch1_1a(x1_0)
        x1_1b = self.branch1_1b(x1_0)
        x1 = torch.cat((x1_1a, x1_1b), 1)

        x2_0 = self.branch2_0(x)
        x2_1 = self.branch2_1(x2_0)
        x2_2 = self.branch2_2(x2_1)
        x2_3a = self.branch2_3a(x2_2)
        x2_3b = self.branch2_3b(x2_2)
        x2 = torch.cat((x2_3a, x2_3b), 1)

        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)
        return out

#############################
## Main Network Definition ##
class InceptionBase(nn.Module):
    """
    Reference: https://arxiv.org/pdf/1602.07261.pdf
    """
    def __init__(self, params, num_classes):
        super(InceptionBase, self).__init__()
        self.features = nn.Sequential(
            BasicConv2d(1 , 32, kernel_size=3, stride=2),
            BasicConv2d(32, 32, kernel_size=3, stride=1),
            BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1),
            Mixed_3a(),
            Mixed_4a(),
            Mixed_5a(),
            Inception_A(),
            Inception_A(),
            Inception_A(),
            Inception_A(),
            Reduction_A(), # Mixed_6a
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Reduction_B(), # Mixed_7a
            Inception_C(),
            Inception_C(),
            Inception_C(),
            nn.AvgPool2d(2, count_include_pad=False)
        )
        self.classif = nn.Linear(1536, num_classes)

    def forward(self, s):
        """
        Args:
            s: (Variable) contains a batch of images, of dimension batch_size x 1 x 128 x params.width

        Returns:
            out: (Variable) dimension batch_size x num_classes
        """
        s = self.features(s)
        s = s.view(s.size(0), -1)
        s = self.classif(s) 
        return s


# Inception-ResNet

######################
## Helper functions ##
class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False) # verify bias false
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001, momentum=0, affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Mixed_5b(nn.Module):

    def __init__(self):
        super(Mixed_5b, self).__init__()

        self.branch0 = BasicConv2d(192, 96, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(192, 48, kernel_size=1, stride=1),
            BasicConv2d(48, 64, kernel_size=5, stride=1, padding=2)
        ) 

        self.branch2 = nn.Sequential(
            BasicConv2d(192, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1),
            BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(192, 64, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out

class Block35(nn.Module):

    def __init__(self, scale=1.0):
        super(Block35, self).__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(320, 32, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(320, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(320, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 48, kernel_size=3, stride=1, padding=1),
            BasicConv2d(48, 64, kernel_size=3, stride=1, padding=1)
        )

        self.conv2d = nn.Conv2d(128, 320, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out

class Mixed_6a(nn.Module):

    def __init__(self):
        super(Mixed_6a, self).__init__()
        
        self.branch0 = BasicConv2d(320, 384, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv2d(320, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out

class Block17(nn.Module):

    def __init__(self, scale=1.0):
        super(Block17, self).__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(1088, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(1088, 128, kernel_size=1, stride=1),
            BasicConv2d(128, 160, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(160, 192, kernel_size=(7,1), stride=1, padding=(3,0))
        )

        self.conv2d = nn.Conv2d(384, 1088, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out

class Mixed_7a(nn.Module):

    def __init__(self):
        super(Mixed_7a, self).__init__()
        
        self.branch0 = nn.Sequential(
            BasicConv2d(1088, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(1088, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 288, kernel_size=3, stride=2)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(1088, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 288, kernel_size=3, stride=1, padding=1),
            BasicConv2d(288, 320, kernel_size=3, stride=2)
        )

        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out

class Block8(nn.Module):

    def __init__(self, scale=1.0, noReLU=False):
        super(Block8, self).__init__()

        self.scale = scale
        self.noReLU = noReLU

        self.branch0 = BasicConv2d(2080, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(2080, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 224, kernel_size=(1,3), stride=1, padding=(0,1)),
            BasicConv2d(224, 256, kernel_size=(3,1), stride=1, padding=(1,0))
        )

        self.conv2d = nn.Conv2d(448, 2080, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out


#############################
## Main Network Definition ##
class InceptionResnetBase(nn.Module):
    """
    Reference: https://arxiv.org/pdf/1602.07261.pdf
    """
    def __init__(self, params, num_classes):
        super(InceptionResnetBase, self).__init__()
        self.conv2d_1a = BasicConv2d(1 , 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80 , kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.maxpool_5a = nn.MaxPool2d(3, stride=2)
        self.mixed_5b = Mixed_5b()
        self.repeat = nn.Sequential(
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17)
        )
        self.mixed_6a = Mixed_6a()
        self.repeat_1 = nn.Sequential(
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            #Block17(scale=0.10),
            #Block17(scale=0.10),
            #Block17(scale=0.10),
            #Block17(scale=0.10),
            #Block17(scale=0.10),
            #Block17(scale=0.10),
            #Block17(scale=0.10),
            #Block17(scale=0.10),
            #Block17(scale=0.10),
            #Block17(scale=0.10),
            #Block17(scale=0.10),
            Block17(scale=0.10)
        )
        self.mixed_7a = Mixed_7a()
        self.repeat_2 = nn.Sequential(
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20)
        )
        self.block8 = Block8(noReLU=True)
        self.conv2d_7b = BasicConv2d(2080, 1536, kernel_size=1, stride=1)
        self.avgpool_1a = nn.AvgPool2d(2, count_include_pad=False)
        self.classif = nn.Linear(1536, num_classes)

    def forward(self, s):
        """
        Args:
            s: (Variable) contains a batch of images, of dimension batch_size x 1 x 128 x params.width

        Returns:
            out: (Variable) dimension batch_size x num_classes
        """
        s = self.conv2d_1a(s)
        s = self.conv2d_2a(s)
        s = self.conv2d_2b(s)
        s = self.maxpool_3a(s)
        s = self.conv2d_3b(s)
        s = self.conv2d_4a(s)
        s = self.maxpool_5a(s)
        s = self.mixed_5b(s)
        s = self.repeat(s)
        s = self.mixed_6a(s)
        s = self.repeat_1(s)
        s = self.mixed_7a(s)
        s = self.repeat_2(s)
        s = self.block8(s)
        s = self.conv2d_7b(s)
        s = self.avgpool_1a(s)
        s = s.view(s.size(0), -1)
        s = self.classif(s) 
        return s


# SqueezeNet
import torch.nn.init as init
from torchvision.models.squeezenet import *

######################
## Helper functions ##
class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

#############################
## Main Network Definition ##
class SqueezeNetBase(nn.Module):
    """
    Modified from official Version 1.1: 
     - https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1
    Documentation for reference: http://pytorch.org/docs/master/nn.html
    """
    def __init__(self, params, num_classes):
        """
        Args:
            params: (Params) contains dropout
        """
        super(SqueezeNetBase, self).__init__()

        self.num_classes = num_classes
        self.dropoutrate = params.dropout
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(64, 16, 64, 64),
            Fire(128, 16, 64, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(128, 32, 128, 128),
            Fire(256, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            Fire(512, 64, 256, 256),
        )
        # Final convolution is initialized differently form the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropoutrate),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AvgPool2d(6)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal(m.weight.data, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, s):
        """
        Args:
            s: (Variable) contains a batch of images, of dimension batch_size x 1 x 128 x 192

        Returns:
            out: (Variable) dimension batch_size x num_classes with the log prob for the labels
        """ 
        s = self.features(s)
        s = self.classifier(s)
        return s.view(s.size(0), self.num_classes)


# ResNet

######################
## Helper functions ##
class Block(nn.Module):
    def __init__(self, in_channels, channels, stride=1, downsample=None, padding=1):
       super(Block, self).__init__()
       self.conv1 = nn.Conv2d(in_channels, channels,3, stride=stride, padding=padding, bias=False)
       self.bn1 = nn.BatchNorm2d(channels)
       self.conv2 = nn.Conv2d(channels, channels,3, stride=1, padding=padding, bias=False)
       self.bn2 = nn.BatchNorm2d(channels)
       self.downsample = downsample
       self.stride = stride

    def forward(self, s):
        residual = s 
        out = self.conv1(s) 
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(s)
        sys.stdout.flush()
        out += residual
        out = F.relu(out)
        return out


#############################
## Main Network Definition ##
class ResNet18(nn.Module):
    """
    Modified from ResNet18
    """
    def __init__(self, params, num_classes):
        """
        Args:
            params: (Params) contains num_channels
        """
        super(ResNet18, self).__init__()
        layers = [2,2,2,2]
        self.num_channels = params.num_channels
        self.inchannels   = params.num_channels
        
        self.conv1 = nn.Conv2d(1,self.inchannels, 3, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inchannels)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1 = self._make_layer(Block,  self.num_channels,layers[0], stride=2)
        self.layer2 = self._make_layer(Block,2*self.num_channels,layers[1], stride=2)
        self.layer3 = self._make_layer(Block,4*self.num_channels,layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(4,stride=1)
        self.fc1 = nn.Linear(2*128*int(np.ceil(params.width/128.))*2, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, channels, num_layers, stride=1):
        downsample = None
        if stride != 1 or self.inchannels != channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.inchannels, channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels),
            )
        layers = []
        layers.append(block(self.inchannels, channels, stride, downsample))
        self.inchannels = channels
        for i in range(1, num_layers):
            layers.append(block(self.inchannels, channels))
        return nn.Sequential(*layers)

    def forward(self, s):
        s = self.conv1(s)
        s = self.bn1(s)
        s = F.relu(s)
        s =  self.maxpool(s)
        s = self.layer1(s)
        s = self.layer2(s)
        s = self.layer3(s)
        s = self.avgpool(s)
        s = s.view(s.size(0),-1)
        return self.fc1(s)






#--------------------------------Multi-label-------------------------------------
def loss_fn(outputs, labels):
    """
    Multi-label loss function
     sigmoid + binary cross entropy loss for better numerical stability
    """
    return nn.BCEWithLogitsLoss()(outputs, labels)

## WARP loss
#-- Weighted Approximate-Rank Pairwise loss
from utils import WARP, WARPLoss

def loss_warp(outputs, labels):
    """
    Sigmoid + WARP loss
    """
    return WARPLoss()(F.sigmoid(outputs),labels)

## LSEP loss
#-- Log-sum-exp-pairwise loss
from utils import LSEP, LSEPLoss

def loss_lsep(outputs, labels):
    """
    Sigmoid + LSEP loss
    """
    return LSEPLoss()(F.sigmoid(outputs),labels)


##########################################

def accuracy(outputs, labels, threshold):
    """
    Compute the accuracy given the outputs and labels for all images.
    Returns: (float) accuracy in [0,1]
    """
    outputs = F.sigmoid(outputs).data.gt(threshold).cpu().numpy()
    labels  = labels.data.cpu().numpy()
    acc = np.mean([met.accuracy_score(labels[i], outputs[i]) for i in range(labels.shape[0])])
    return acc

def precision(outputs, labels, threshold):
    """
    Compute the precision given the outputs and labels for all images.
    Returns: (float) accuracy in [0,1]
    """
    outputs = F.sigmoid(outputs).data.gt(threshold).cpu().numpy()
    labels  = labels.data.cpu().numpy()
    prec = np.mean([met.precision_score(labels[i], outputs[i]) for i in range(labels.shape[0])])
    return prec

def recall(outputs, labels, threshold):
    """
    Compute the recall given the outputs and labels for all images.
    Returns: (float) accuracy in [0,1]
    """
    outputs = F.sigmoid(outputs).data.gt(threshold).cpu().numpy()
    labels  = labels.data.cpu().numpy()
    rec = np.mean([met.recall_score(labels[i], outputs[i]) for i in range(labels.shape[0])])
    return rec

def f1(outputs, labels, threshold):
    """
    Compute the F1 (harmonic mean of precision and recall) given the outputs and labels for all images.
    Returns: (float) accuracy in [0,1]
    """
    outputs = F.sigmoid(outputs).data.gt(threshold).cpu().numpy()
    labels  = labels.data.cpu().numpy()
    rec = np.mean([met.f1_score(labels[i], outputs[i]) for i in range(labels.shape[0])])
    return rec

## metrics
metrics = {
    'accuracy' : accuracy,
    'precision': precision,
    'recall'   : recall,
    'f1'       : f1,
}

#--------------------------------Single-label-------------------------------------
def loss_fn_sing(outputs, labels):
    """
    Single-label loss function
     (log) softmax + binary cross entropy loss for better numerical stability
    """
    return nn.CrossEntropyLoss()(outputs, labels.long())

def accuracy_sing(outputs, labels, threshold):
    """
    Compute the accuracy given the outputs and labels for all images.
    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(F.softmax(outputs, dim=1).data.cpu().numpy(), axis=1)
    labels  = labels.data.cpu().numpy() # LongTensor not one-hot
    acc = np.sum(outputs==labels)/float(labels.size)
    return acc

## metrics
metrics_sing = {
    'accuracy' : accuracy_sing,
}
