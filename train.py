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
import sys

LEARNING_RATE_G = 0.002
LEARNING_RATE_D = 0.002

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

########################
# Model initialization #
########################
#if need_pretraining == 1, train the generator
try:
    need_pretraining = int(sys.argv[1])
except:
    need_pretraining = 0

G = generator.generator_nn(3,3)
if need_pretraining == 0:
        G.load_state_dict(torch.load("pretrained_G.pt"))
D = discriminator.discriminator_nn(3,1)
G.train()
D.train()
VGG_model = get_vgg19(16, True, "vgg19-dcbb9e9d.pth")
BCE_loss = nn.BCELoss()
L1_loss = nn.L1Loss()
G_optim = optims.Adam(G.parameters(), lr = LEARNING_RATE_G)
D_optim = optims.Adam(D.parameters(), lr = LEARNING_RATE_D)


##############################
# Model pretraining with VGG #
##############################
if need_pretraining == 1:
    anime_dataset = load_training_set("training_set/nonfiure_anime_totoro")
    #target
    print(len(anime_dataset))
    for epoch in range(10):
        for batch_idx, (data, target) in enumerate(anime_dataset):
            G_optimizer.zero_grad()
            x_val = VGG_model(data)
            G_val = VGG_model(G(data))
            loss = L1_loss(G_val, x_val)
            loss.backward()
            G_optim.step()

        print("model_pretrained for epoch",epoch,"now saving it")
        torch.save(G.state_dict_dict(), "pretrained_G.pt")


# put this part before pre-training model

train_anime = load_training_set('nonfiure path')
train_real_scenery = load_training_set('real_scenery path')

# put this part of code into the main function

# initialize zero and one matrix
real = torch.ones(batch_size, 1, input_size // 4, input_size // 4)
fake = torch.zeros(batch_size, 1, input_size // 4, input_size // 4)

for epoch in range(num_epoch):
    # pre-train data
    G.train()

    # train discriminator
    # zip(iterator) returns an interator of tuples
    for (x, _), (y, _) in zip(train_anime, train_real_scenery):
        # x : anime input
        # y : real_scenery input (center)
        # e : real_scenery input (edge)
        e = y[:, :, :, args.input_size:]
        y = y[:, :, :, :args.input_size]

        # 1. train discriminator D
        # initialize gradients
        D_optimizer.zero_grad()

        # train on real
        d_real = D(y)
        dr_loss = BCE_loss(d_real, real)

        # train on fake
        d_fake = D(G(x))
        df_loss = BCE_loss(d_fake, fake)

        # train on edge
        d_edge = D(e)
        de_loss = BCE_loss(d_edge, fake)

        # sum up loss function to dicriminator loss
        D_loss = dr_loss + df_loss + de_loss
        D_loss.backward()
        d_optimizer.step()

        # 2. train generator G
        G_optmizer.zero_grad()

        # adverserial loss
        d_fake = D(G(x))
        adv_loss = BCE_loss(d_fake, real)

        # content loss (sth i dont know at all)
        x_feature = VGG((x + 1) / 2)
        G_feature = VGG((G(x) + 1) / 2)
        con_loss = L1_loss(G_feature, x_feature)

        # sum up generator loss function
        G_loss = adv_loss + con_loss
        G_loss.backward()
        G_optimizer.step()

# save parameters of G and D
torch.save(G.state_dict(), 'generator_param_v1.pkl')
torch.save(D.state_dict(), 'discriminator_param_v1.pkl')
