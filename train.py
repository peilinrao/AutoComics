import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.autograd as autograd
import torch.optim as optims
import numpy as np
from torch.autograd import Variable
import models
import models.generator as generator
import models.discriminator as discriminator
import sys


LEARNING_RATE_G = 0.002
LEARNING_RATE_D = 0.002
BATCH_SIZE = 8
num_epoch_pretrain = 10
num_epoch_train = 100
########################
# Model initialization #
########################
#if need_pretraining == 1, train the generator
try:
    need_pretraining = int(sys.argv[1])
except:
    # By default we assume that there is a pretrained model
    need_pretraining = 0

G = generator.generator_nn(3,3)
G.load_state_dict(torch.load("param/pretrained_G.pt"))
print("Pretrained model loaded!")
D = discriminator.discriminator_nn(3,1)
G.train()
D.train()
VGG_model = get_vgg19(16, True, "utils/vgg19-dcbb9e9d.pth")
BCE_loss = nn.BCELoss()
L1_loss = nn.L1Loss()
G_optim = optims.Adam(G.parameters(), lr = LEARNING_RATE_G)
D_optim = optims.Adam(D.parameters(), lr = LEARNING_RATE_D)


##################
# Model Training #
##################
anime_dataset = load_training_set("data_test/nonfigure_anime_test")
train_real_scenery = load_training_set('data_test/nonfigure_realworld_test')

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
        print("batch info:",count)
        count+=1

        print("starting training")
        try:
            D_optim.zero_grad()
            d_real = D(y)
            dr_loss = BCE_loss(d_real, real)

            d_fake = D(G(x))
            df_loss = BCE_loss(d_fake, fake)

            D_loss = dr_loss + df_loss
            D_loss.backward()
            D_optim.step()

            # train generator G
            G_optim.zero_grad()

            # adverserial loss
            d_fake = D(G(x))
            adv_loss = BCE_loss(d_fake, real)

            # content loss
            x_feature = VGG_model(x)
            G_feature = VGG_model(G(x))
            con_loss = L1_loss(G_feature, x_feature)

            # sum up generator loss function
            G_loss = adv_loss + con_loss
            G_loss.backward()
            G_optim.step()
        except:
            continue

    # save parameters of G and D
    torch.save(G.state_dict(), 'param/generator_param.pt')
    torch.save(D.state_dict(), 'param/discriminator_param.pt')
