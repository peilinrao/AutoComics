import torch
import utils
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
from torch.autograd import Variable

class generator_nn(nn.Module):
    def __init__():
        super(generator_nn, self).__init__()
        num_feature = 64
        down_convolution_seq = nn.Sequential(
          nn.Conv2d(in_channel, num_feature, kernel_size=7, stride=1,padding = 1),
          nn.InstanceNorm2d(num_feature),
          nn.ReLU(),
          nn.Conv2d(num_feature, num_feature*2, kernel_size=3, stride=2,padding = 1),
          nn.Conv2d(num_feature*2, num_feature*2, kernel_size=3, stride=1,padding = 1),
          nn.ReLU()
          nn.Conv2d(num_feature*2, num_feature*4, kernel_size=3, stride=2,padding = 1),
          nn.Conv2d(num_feature*4, num_feature*4, kernel_size=3, stride=1,padding = 1),
          nn.InstanceNorm2d(num_feature*4),
          nn.ReLU()
        )

        self.resnet
