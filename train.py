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


LEARNING_RATE_G == 0.002
LEARNING_RATE_D == 0.002

def get_vgg19(nf, pretrained,path):
    net= models.vgg19()
    if pretrained:
        net.load_state_dict(torch.load(path))
    net.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096,4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, nf),
    )
    return net

Generator_model = generator.generator_nn(3,3)
Discriminator_model = discriminator.discriminator_nn(3,1)
BCE = nn.BCELoss()
L1_loss == nn.L1loss()
Generator_optimizer = optim.Adam(Generator_model.parameters(), lr = LEARNING_RATE_G, (0.5, 0.999))
Discriminator_model = optim.Adam(Discriminator_model.parameters(), lr = LEARNING_RATE_D, (0.5, 0.999))


print(get_vgg19(16, True, "vgg19-dcbb9e9d.pth"))
