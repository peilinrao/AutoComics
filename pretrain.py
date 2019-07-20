import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.autograd as autograd
import torch.optim as optims
import numpy as np
from torch.autograd import Variable
import models
import sys

LEARNING_RATE_G = 0.002
LEARNING_RATE_D = 0.002
IMAGE_SIZE = 25
num_epoch_pretrain = 10
########################
# Model initialization #
########################
if torch.cuda.is_available():
    print("Using GPU...")
else:
    print("Using CPU")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

G = models.generator_nn(3,3)
G.to(device)
G.train()
# Download vgg19 at https://download.pytorch.org/models/vgg19-dcbb9e9d.pth
VGG_model = models.get_vgg19(64, True, "utils/vgg19-dcbb9e9d.pth")
VGG_model.to(device)
VGG_model.eval()
L1_loss = nn.L1Loss().to(device)
G_optim = optims.Adam(G.parameters(), lr = LEARNING_RATE_G)


####################
# Model pretraining#
####################

print("Start pretraining")
anime_dataset = models.load_training_set("data_test/nonfigure_anime_test")
for epoch in range(num_epoch_pretrain):
    print("epoch",epoch,"/",num_epoch_pretrain)
    for batch_idx, (data, target) in enumerate(anime_dataset):
        print("Starting batch",batch_idx, "/",len(anime_dataset))
        data = data.to(device)
        G_optim.zero_grad()
        x_val = VGG_model(data)
        G_val = VGG_model(G(data))
        loss = L1_loss(G_val, x_val)
        print("loss:",loss)
        loss.backward()
        G_optim.step()

    print("model pretrained for epoch",epoch,"now saving it")
    torch.save(G.state_dict(), "param/pretrained_G.pt")
