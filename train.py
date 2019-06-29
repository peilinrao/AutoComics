import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.autograd as autograd
import torch.optim as optims
import numpy as np
from torch.autograd import Variable
import generator
import discriminator
import helpers

LEARNING_RATE_G = 0.002
LEARNING_RATE_D = 0.002

#################
#Helper functions
#################
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
        nn.Linear(4096, nf),
    )
    return net



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
            torchvision.transforms.Resize((256,256)),
            torchvision.transforms.ToTensor()
            ])
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=8,
        num_workers=0,
        shuffle=True
    )
    return train_loader

#################
# Model building
#################
G = generator.generator_nn(3,3)
D = discriminator.discriminator_nn(3,1)
G.train()
D.train()
VGG_model = get_vgg19(16, True, "vgg19-dcbb9e9d.pth")
BCE_loss = nn.BCELoss()
L1_loss = nn.L1Loss()
Generator_optimizer = optims.Adam(G.parameters(), lr = LEARNING_RATE_G, betas = (0.5, 0.999))
Discriminator_optimizer = optims.Adam(D.parameters(), lr = LEARNING_RATE_D, betas = (0.5, 0.999))


#################
# Testing field
#################

#################
# Model running
#################

anime_dataset = load_training_set("training_set")
#target
print(len(anime_dataset))
for epoch in range(10):
    for batch_idx, (data, target) in enumerate(anime_dataset):
        G_optimizer.zero_grad()
        x_val = VGG_model(data)
        G_val = VGG_model(G(data))

        loss = L1_loss(G_val, x_val)

        break
