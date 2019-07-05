import torch
import torch.nn as nn
import torch.nn.functional as F

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


# Set up discriminator model
class discriminator_nn(nn.Module):
    # initializers
    def __init__(self, in_chn, out_chn, n=16):
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

        helpers.initialize_weights(self)

    # forward method
    def forward(self, input):
        output = self.conv(input)
        return output


# Set up helper for generator model
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
            nn.InstanceNorm2d(nf),
        )

        helpers.initialize_weights(self)

    def forward(self, input):
        output = input + self.conv(input)

        return output

# Set up generator model
class generator_nn(nn.Module):
    def __init__(self, in_chn, out_chn, nf=16, nb=8):
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

        # check this alternative code works
        # self.resnet = nn.Sequential([resnet_block(nf * 4, kernel_size=3, stride=1, padding=1) for i range(nb)])

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

        helpers.initialize_weights(self)

    def forward(self, input):
        temp = self.down_conv(input)
        temp = self.resnet(temp)
        output = self.up_conv(temp)

        return output
