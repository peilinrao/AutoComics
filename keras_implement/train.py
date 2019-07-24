import numpy as np
import time, os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, ZeroPadding2D
from keras.layers import ReLU, LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.layers import Add
from keras.optimizers import Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator

class CartoonGAN(object):
    def __init__(self, args):
        self.args = args
        self.D = None   # discriminator
        self.G = None   # generator
        self.DM = None  # discriminator models
        self.AM = None  # adversarial models

    def discriminator(self):
        # discriminator returns a scalar: input's label
        if self.D:
            return self.D
        batch_size = self.args['batch_size']    # 8
        ndf = self.args['ndfeature']            # 32
        row = self.args['row']                  # 256
        col = self.args['col']                  # 256
        alpha = 0.2
        dropout = 0.5
        momentum = 0.1
        # ====== discriminator model ======
        # output size: O = (W-K+2P)/S+1
        D = Sequential()
        # input size 1: 256 x 256 x 3
        input_shape = (batch_size, row, col, 3)
        D.add(Conv2D(ndf * 1, kernel_size=3, strides=1, input_shape=input_shape, data_format='channels_last'))
        D.add(LeakyReLU(alpha))
        D.add(Dropout(dropout))
        # state size 2: 128 x 128 x ndf
        D.add(Conv2D(ndf * 2, kernel_size=3, strides=2, padding='same'))
        D.add(LeakyReLU(alpha))
        D.add(Dropout(dropout))
        # state size 3: 64 x 64 x ndf*2
        D.add(Conv2D(ndf * 4, kernel_size=3, strides=1))
        D.add(BatchNormalization(momentum=momentum))
        D.add(LeakyReLU(alpha))
        D.add(Dropout(dropout))
        # state size 4: ...
        D.add(Conv2D(ndf * 4, kernel_size=3, strides=2, padding='same'))
        D.add(LeakyReLU(alpha))
        D.add(Dropout(dropout))
        # state size 5: ...
        D.add(Conv2D(ndf * 8, kernel_size=3, strides=1))
        D.add(BatchNormalization(momentum=momentum))
        D.add(LeakyReLU(alpha))
        D.add(Dropout(dropout))
        # state size 6: ...
        D.add(Conv2D(ndf * 8, kernel_size=3, strides=1))
        D.add(BatchNormalization(momentum=momentum))
        D.add(LeakyReLU(alpha))
        D.add(Dropout(dropout))
        # state size 7: ...
        D.add(Conv2D(1, kernel_size=3, strides=1))
        # single output layer
        D.add(Flatten())
        D.add(Dense(1, activation='sigmoid'))
        D.summary()
        self.D = D
        return self.D

    def generator(self):
        # generator returns a 256 x 256 x 3 image
        if self.G:
            return self.G
        batch_size = self.args['batch_size']    # 8
        ngf = self.args['ngfeature']            # 64
        nblock = self.args['nblock']            # 8
        row = self.args['row']                  # 256
        col = self.args['col']                  # 256
        alpha = 0.0
        dropout = 0.5
        momentum = 0.1
        # ====== generator model ======
        # flat convolution
        G = Sequential()
        # input size 1: 256 x 256 x 3
        input_shape = (batch_size, row, col, 3)
        G.add(Conv2D(ngf, kernel_size=7, strides=1, input_shape=input_shape, data_format='channels_last'))
        G.add(BatchNormalization(momentum=momentum))
        G.add(LeakyReLU(alpha))
        G.add(Dropout(dropout))
        # state size 2: 256 x 256 x ngf
        # down convolution
        G.add(Conv2D(ngf*2, kernel_size=3, strides=2, padding='same'))
        G.add(Conv2D(ngf*2, kernel_size=3, strides=1))
        G.add(BatchNormalization(momentum=momentum))
        G.add(LeakyReLU(alpha))
        G.add(Dropout(dropout))
        # state size 3: 128 x 128 x ngf*2
        G.add(Conv2D(ngf*4, kernel_size=3, strides=2, padding='same'))
        G.add(Conv2D(ngf*4, kernel_size=3, strides=1))
        G.add(BatchNormalization(momentum=momentum))
        G.add(LeakyReLU(alpha))
        G.add(Dropout(dropout))
        # state size 4: 64 x 64 x ngf*4
        # resnet blocks convolution
        last = None
        for i in range(nblock):
            R, resblock = self.resnet_block(last)
            last = R
            G.add(resblock)
            G.add(Dropout(dropout))
        # up convolution
        # state size 1 (from resblocks): 64 x 64 x ngf*4
        G.add(Conv2DTranspose(ngf*2, kernel_size=3, strides=2, padding='same'))
        G.add(Conv2DTranspose(ngf*2, kernel_size=3, strides=1))
        G.add(BatchNormalization(momentum=momentum))
        G.add(LeakyReLU(alpha))
        G.add(Dropout(dropout))
        # state size 2: 128 x 128 x ngf*2
        G.add(Conv2DTranspose(ngf, kernel_size=3, strides=2, padding='same'))
        G.add(Conv2DTranspose(ngf, kernel_size=3, strides=1))
        G.add(BatchNormalization(momentum=momentum))
        G.add(LeakyReLU(alpha))
        G.add(Dropout(dropout))
        # state size 3: 256 x 256 x ngf
        G.add(Conv2D(3, kernel_size=7, strides=1))
        # output state: 256 x 256 x 3
        G.summary()
        self.G = G
        return self.G

    def resnet_block(self, last):
        nf = self.args['ngfeature']*4   # 256
        alpha = 0.0
        momentum = 0.1
        # ====== resnet block ======
        R = Sequential()
        R.add(Conv2D(nf, kernel_size=3, strides=1))
        R.add(BatchNormalization(momentum=momentum))
        R.add(LeakyReLU(alpha))
        R.add(Conv2D(nf, kernel_size=3, strides=1))
        R.add(BatchNormalization(momentum=momentum))
        resblock = Sequential()
        resblock.add(Add()[last, R])
        return R, resblock

    def discriminator_model(self):
        if self.DM:
            return self.DM
        lr = self.args['lrD']              # 1e-5
        beta1 = 0.5
        lr_decay = 5e-8
        optim = Adam(lr=lr, decay=lr_decay)
        # ====== discriminator model ======
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy'])
        return self.DM

    def adversarial_model(self):
        if self.AM:
            return self.AM
        lr = self.args['lrG']           # 1e-5
        beta1 = 0.5
        lr_decay = 3e-8
        optim = Adam(lr=lr, decay=lr_decay)
        # ====== adversarial model ======
        self.AM = Sequential()
        self.AM.add(self.generator())
        self.AM.add(self.discriminator())
        self.AM.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy'])
        return self.AM

class model():
    def __init__(self, args):
        self.args = args
        # ====== CartoonGAN model ======
        self.CartoonGAN = CartoonGAN(args)
        self.discriminator = self.CartoonGAN.discriminator_model()
        self.adversarial = self.CartoonGAN.adversarial_model()
        self.generator = self.CartoonGAN.generator()

    def train(self):
        log = '------ start training model ------'
        print(log)
        # ====== train model ======
        nepoch = self.args['nepoch']
        nbatch = self.args['nbatch']
        batch_size = self.args['batch_size']
        for epoch in range(nepoch):
            log = 'epoch: %d / %d' % (epoch, nepoch)
            print(log)
            for batch in range(nbatch):
                # ====== train discriminator ======
                real, noise = self.get_train_batch()
                fake = self.generator.predict(noise)
                x = np.concatenate((real, fake))
                y = np.concatenate((np.ones([batch_size, 1]), np.zeros([batch_size, 1])))
                d_loss = self.discriminator.train_on_batch(x,y)
                # ====== train adversarial ======
                y = np.ones([batch_size, 1])
                a_loss = self.adversarial.train_on_batch(noise, y)
                # ====== log message ======
                log = 'batch %d / %d\n' % (batch, nbatch)
                log += 'discriminator loss: %d\n adversarial loss: %d' % (d_loss, a_loss)
                print(log)
            # test function
            # self.print_weights(self.generator)
        log = '------ finish training ------'
        print(log)
        log = '------ save model ------'
        print(log)
        # ====== save model ======
        self.save(self.generator)

    def test(self, plot_sample=True):
        test = get_test(self)
        result = self.generator.predict()
        if plot_sample:
            image = result[0]
            self.plot_image(image)

    def get_train_batch(self):
        cartoon_dir = self.args['cartoon_dir']
        photo_dir = self.args['photo_dir']
        row = self.args['row']
        col = self.args['col']
        batch_size = self.args['batch_size']
        datagen = ImageDataGenerator(
            rescale = 1./255,
            data_format = 'channels_last'
        )
        cartoon_batch = datagen.flow_from_directory(
            directory = cartoon_dir,
            target_size = (row, col),
            batch_size = batch_size,
            shuffle = True,
            class_mode = None
        )
        photo_batch = datagen.flow_from_directory(
            directory = photo_dir,
            target_size = (row, col),
            batch_size = batch_size,
            shuffle = True,
            class_mode = None
        )

        return cartoon_batch, photo_batch

    def get_test(self):
        test_dir = self.args['test_dir']
        row = self.args['row']
        col = self.args['col']
        batch_size = len([f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))])
        datagen = ImageDataGenerator(
            rescale = 1./255,
            data_format = 'channels_last'
        )
        test = datagen.flow_from_directory(
            directory = test_dir,
            target_size = (row, col),
            batch_size = batch_size,
            shuffle = False,
            class_mode = None
        )
        return test

    def plot_image(self, image):
        plt.imshow(image)
        plt.show()

    def save(self, model):
        return

def arguments():
    args = {
    # ====== data level ======
        'cartoon_dir' : '',
        'photo_dir' : '',
        'test_dir' : '',
        'save_param' : None,
    # ====== train level ======
        'nepoch_pretrain' : 10,
        'nepoch' : 100,
        'nbatch' : 100,
        'batch_size' : 32,
    # ====== model level ======
        'row' : 256,
        'col' : 256,
        'ndfeature' : 32,
        'ngfeature' : 64,
        'nblock' : 8,
        'nblock' : 8,
        'lrD' : 1e-5,
        'lrG' : 1e-5
    }
return args

# ====== main method ======
args = arguments()
model = model(args)
model.train()
model.test()
