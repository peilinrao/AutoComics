import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import torchvision.models as models
import numpy as np
from torch.autograd import Variable
import generator
import discriminator


Generator = generator.generator_nn(3,3)
Discriminator = discriminator.discriminator(3,1)

def get_vgg19(nf, pretrained):
    model = models.vgg19()
    if pretrained:
        net.load_state_dict(torch.load(pretrained_vgg19_path))
    net.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096,4096),
        nn.RelU(True),
        nn.Dropout(),
        nn.Linear(4096, nf),
    )
    return net
