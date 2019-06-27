import torch
import utils
import torch.nn as nn
import torch.nn.functional as F

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

        utils.initialize_weights(self)

    def forward(self, input):
        output = input + self.conv(input)

        return output

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
          nn.Conv2d(in_chn, nf, kernel_size=7, stride=1, padding = 1),
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

        utils.initialize_weights(self)

    def forward(self, input):
        temp = self.down_conv(input)
        temp = self.resnet(temp)
        output = self.up_conv(temp)

        return output
