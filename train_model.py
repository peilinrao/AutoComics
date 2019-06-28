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
        G_feature = VGG((G_ + 1) / 2)
        con_loss = args.con_lambda * L1_loss(G_feature, x_feature.detach())

        # sum up generator loss function
        G_loss = adv_loss + con_loss
        G_loss.backward()
        G_optimizer.step()

# save parameters of G and D
torch.save(G.state_dict(), 'generator_param_v1.pkl')
torch.save(D.state_dict(), 'discriminator_param_v1.pkl')
        
