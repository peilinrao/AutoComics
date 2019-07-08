import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.autograd as autograd
import torch.optim as optims
import numpy as np
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
import models
import sys


# See how the trained network perform
G = generator.generator_nn(3,3)
G.load_state_dict(torch.load("param/generator_param.pt"))

image = Image.open("results/coast_test.jpg")
torchvision.transforms.Resize((256,256))
image = ToTensor()(image).unsqueeze(0)
image = G(image).squeeze(0)
plt.imshow(ToPILImage()(image))
plt.show()
