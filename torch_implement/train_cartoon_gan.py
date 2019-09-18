import sys
sys.path.append('model')

import torch.nn as nn
import arguments
from cartoon_gan import CartoonGAN
import data_loader

# main training
model = CartoonGAN(arguments)
model.initialize()

data_loader = data_loader()

num_batch = floor(min(data_loader.size(), args['ntrain']) / args['batch_size'])

print('-------------start training------------------')
for epoch in range(args['niter']+args['niter_decay']):
    print('epoch: %d / %d' % (epoch, args['niter']+args['niter_decay']))
    ###
    # sth

    for batch in range(num_batch):
        print('batch: %d / %d' % (batch, num_batch))
        ... = data_loader.get_next_batch()
        model.forward(...)
        model.optimize_param(...)

        # print loss

    # save latest model
    model.save(...)

    if epoch > args['niter']:
        # update learning rate

    model.refresh_param()
