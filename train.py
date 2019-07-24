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
BATCH_SIZE = 4
num_epoch_pretrain = 10
num_epoch_train = 200
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
G.load_state_dict(torch.load("param/pretrained_G.pt"))
print("Pretrained model loaded!")
D = models.discriminator_nn(3,1)
D.to(device)
G.train()
D.train()
VGG_model = models.get_vgg19(64, True, "utils/vgg19-dcbb9e9d.pth")
VGG_model.to(device)
VGG_model.eval()
BCE_loss = nn.BCELoss().to(device)
L1_loss = nn.L1Loss().to(device)
G_optim = optims.Adam(G.parameters(), lr = LEARNING_RATE_G, betas=(0.5, 0.999))
D_optim = optims.Adam(D.parameters(), lr = LEARNING_RATE_D, betas=(0.5, 0.999))

G_scheduler = optims.lr_scheduler.MultiStepLR(optimizer=G_optim, milestones=[num_epoch_train // 2, num_epoch_train // 4 * 3], gamma=0.1)
D_scheduler = optims.lr_scheduler.MultiStepLR(optimizer=D_optim, milestones=[num_epoch_train // 2, num_epoch_train // 4 * 3], gamma=0.1)


##############################
# Model pretraining with VGG #
##############################
if need_pretraining == 1:
    print("pretraining")
    anime_dataset = load_training_set("data_test/nonfigure_anime_test")
    for epoch in range(num_epoch_pretrain):
        print("epoch",epoch,"/",num_epoch_pretrain)
        for batch_idx, (data, target) in enumerate(anime_dataset):
            print("Starting batch",batch_idx, "/",len(anime_dataset))
            G_optim.zero_grad()
            x_val = VGG_model(data)
            G_val = VGG_model(G(data))
            loss = L1_loss(G_val, x_val)
            loss.backward()
            G_optim.step()

        print("model pretrained for epoch",epoch,"now saving it")
        torch.save(G.state_dict(), "param/pretrained_G.pt")


#######################
# Main Training Model #
#######################

anime_dataset = load_training_set("data_test/nonfigure_anime_test")
train_real_scenery = load_training_set('data_test/nonfigure_realworld_test')

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
        G_scheduler.step()
        D_scheduler.step()
        count+=1
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
