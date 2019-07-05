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
from tqdm import tqdm

LEARNING_RATE_G = 0.002
LEARNING_RATE_D = 0.002
IMAGE_SIZE = 256
BATCH_SIZE = 8
num_epoch_pretrain = 10
num_epoch_train = 100

####################
# Helper functions #
####################
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

########################
# Model initialization #
########################

G = generator.generator_nn(3,3)
D = discriminator.discriminator_nn(3,1)
G.load_state_dict(torch.load("generator_param.pt"))
D.load_state_dict(torch.load("discriminator_param.pt"))
G.train()
D.train()
VGG_model = get_vgg19(16, True, "vgg19-dcbb9e9d.pth")
BCE_loss = nn.BCELoss()
L1_loss = nn.L1Loss()
G_optim = optims.Adam(G.parameters(), lr = LEARNING_RATE_G)
D_optim = optims.Adam(D.parameters(), lr = LEARNING_RATE_D)


##################
# Model Training #
##################
anime_dataset = load_training_set("training_test/nonfigure_anime_test")
train_real_scenery = load_training_set('training_test/nonfigure_realworld_test')

# put this part of code into the main function

# initialize zero and one matrix
real = torch.ones(BATCH_SIZE, 1, 64, 64)
fake = torch.zeros(BATCH_SIZE, 1, 64, 64)
print("model training begins")
for epoch in range(num_epoch_train):
    # pre-train data
    G.train()

    # train discriminator
    # zip(iterator) returns an interator of tuples
    print("epoch info:",epoch,"/",num_epoch_train)
    count = 0
    for (x, _), (y, _) in zip(anime_dataset, train_real_scenery):
        print("batch num:",count)
        count+=1
        # 1. train discriminator D
        # initialize gradients
        print("starting training")
        try:
            D_optim.zero_grad()
            d_real = D(y)
            dr_loss = BCE_loss(d_real, real)

            # train on fake
            d_fake = D(G(x))
            df_loss = BCE_loss(d_fake, fake)

            # sum up loss function to dicriminator loss
            D_loss = dr_loss + df_loss
            D_loss.backward()
            D_optim.step()

            # 2. train generator G
            G_optim.zero_grad()

            # adverserial loss
            d_fake = D(G(x))
            adv_loss = BCE_loss(d_fake, real)

            # content loss (sth i dont know at all)
            x_feature = VGG_model((x + 1) / 2)
            G_feature = VGG_model((G(x) + 1) / 2)
            con_loss = L1_loss(G_feature, x_feature)

            # sum up generator loss function
            G_loss = adv_loss + con_loss
            G_loss.backward()
            G_optim.step()

            print("adv loss:",adv_loss)
            print("con loss:", con_loss)
            print("G loss:", G_loss)
        except:
            print("Some exception happened, but we ignore that")
            continue

    # save parameters of G and D
    torch.save(G.state_dict(), 'generator_param.pt')
    torch.save(D.state_dict(), 'discriminator_param.pt')
