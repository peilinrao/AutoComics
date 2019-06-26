import torch
import torch.nn as nn

class discriminator(nn.Module):
    # initializers
    def __init__(self, in_chn, out_chn, n=32):
        super(discriminator, self).__init__()
        self.input_channel = in_size
        self.output_channel = out_size
        self.layer_input_size = n
        self.conv = nn.Sequential(            
            nn.Conv2d(in_chn, n * 1, kernal_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(n * 1, n * 2, kernal_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(n * 2, n * 4, kernal_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(n * 4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(n * 4, n * 4, kernal_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(n * 4, n * 8, kernal_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(n * 4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(n * 8, n * 8, kernal_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(n * 4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(n * 8, out_chn, kernal_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        utils.initialize_weights(self)

    # forward method
    def forward(self, input):
        output = self.conv(input)
        return output
