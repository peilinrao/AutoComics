import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.autograd as autograd
import torch.optim as optims
import numpy as np
from torch.autograd import Variable

BATCH_SIZE = 4
IMAGE_SIZE = 256
####################
# Helper functions #
####################
print("models loaded")
# load_training_set():
# Example:
#    for batch_idx, (data, target) in enumerate(anime_dataset):
#           batch_idx = whicj batch you are dealing
#           data 64 * 3 * 693 * 1280 tensor
#           target = list of size 64, tell whether 0 or 1, figure/nonfigure
def load_training_set(data_path):
    train_dataset = torchvision.datasets.ImageFolder(
        root = data_path,
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
            torchvision.transforms.ToTensor()
            ])
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        shuffle=True
    )
    return train_loader

# initialize weights of layers
def initialize_weights(networks):
    for m in networks.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

def get_vgg19(nf, pretrained,path):
    net= torchvision.models.vgg19()
    if pretrained:
        net.load_state_dict(torch.load(path))
    net.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096,4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, nf)
    )
    return net


#######################
# discriminator model #
#######################
class discriminator_nn(nn.Module):
    # initializers
    def __init__(self, in_chn, out_chn, n=64):
        super(discriminator_nn, self).__init__()
        self.input_channel = in_chn
        self.output_channel = out_chn
        self.layer_input_size = n
        self.conv = nn.Sequential(
            nn.Conv2d(in_chn, n * 1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(n * 1, n * 2, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(n * 2, n * 4, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(n * 4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(n * 4, n * 4, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(n * 4, n * 8, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(n * 4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(n * 8, n * 8, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(n * 4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(n * 8, out_chn, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        initialize_weights(self)

    # forward method
    def forward(self, input):
        output = self.conv(input)
        return output

###################
# generator model #
###################
# Set up resnet_block for generator model
class resnet_block(nn.Module):
    def __init__(self, nf, kernel_size, stride, padding):
        super(resnet_block, self).__init__()
        self.channel = nf
        self.kernel = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv = nn.Sequential(
            nn.Conv2d(nf, nf, kernel_size, stride, padding),
            nn.InstanceNorm2d(nf),
            nn.ReLU(True),
            nn.Conv2d(nf, nf, kernel_size, stride, padding),
            nn.InstanceNorm2d(nf)
        )

        initialize_weights(self)

    def forward(self, input):
        output = input + self.conv(input)

        return output

# Set up generator model
class generator_nn(nn.Module):
    def __init__(self, in_chn, out_chn, nf=64, nb=8):
        # parameters
        super(generator_nn, self).__init__()
        self.input_channel = in_chn
        self.output_channel = out_chn
        self.feature_num = nf
        self.resnet_block_num = nb

        # down-convolution
        self.down_conv = nn.Sequential(
          nn.Conv2d(in_chn, nf, kernel_size=7, stride=1, padding = 3),
          nn.InstanceNorm2d(nf),
          nn.ReLU(True),
          nn.Conv2d(nf * 1, nf * 2, kernel_size=3, stride=2, padding = 1),
          nn.Conv2d(nf * 2, nf * 2, kernel_size=3, stride=1, padding = 1),
          nn.ReLU(True),
          nn.Conv2d(nf * 2, nf * 4, kernel_size=3, stride=2, padding = 1),
          nn.Conv2d(nf * 4, nf * 4, kernel_size=3, stride=1, padding = 1),
          nn.InstanceNorm2d(nf * 4),
          nn.ReLU(True)
        )

        # 8 residual blocks
        self.resnet_blocks = []
        for i in range(nb):
            self.resnet_blocks.append(resnet_block(nf * 4, kernel_size=3, stride=1, padding=1))

        self.resnet = nn.Sequential(*self.resnet_blocks)

        # up-convolution
        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(nf * 4, nf * 2, 3, 2, 1, 1),
            nn.Conv2d(nf * 2, nf * 2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(nf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(nf * 2, nf, 3, 2, 1, 1),
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(nf),
            nn.ReLU(True),
            nn.Conv2d(nf, out_chn, kernel_size=7, stride=1, padding=3),
            nn.Tanh(),
        )

        initialize_weights(self)

    def forward(self, input):
        temp = self.down_conv(input)
        temp = self.resnet(temp)
        output = self.up_conv(temp)

        return output
