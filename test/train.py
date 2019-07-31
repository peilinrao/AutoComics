import numpy as np
import time, os
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, ZeroPadding2D
from keras.layers import ReLU, LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.layers import Add
from keras.optimizers import Adam, RMSprop
import keras.backend as K
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
        input_shape = (row, col, 3)
        D.add(Conv2D(ndf * 1, kernel_size=3, strides=1, padding='same', input_shape=input_shape, name='discriminator_1', data_format='channels_last'))
        D.add(LeakyReLU(alpha))
        D.add(Dropout(dropout))
        # state size 2: 128 x 128 x ndf
        D.add(Conv2D(ndf * 2, kernel_size=3, strides=2, padding='same'))
        D.add(LeakyReLU(alpha))
        D.add(Dropout(dropout))
        # state size 3: 64 x 64 x ndf*2
        D.add(Conv2D(ndf * 4, kernel_size=3, strides=1, padding='same'))
        D.add(BatchNormalization(momentum=momentum))
        D.add(LeakyReLU(alpha))
        D.add(Dropout(dropout))
        # state size 4: ...
        D.add(Conv2D(ndf * 4, kernel_size=3, strides=2, padding='same'))
        D.add(LeakyReLU(alpha))
        D.add(Dropout(dropout))
        # state size 5: ...
        D.add(Conv2D(ndf * 8, kernel_size=3, strides=1, padding='same'))
        D.add(BatchNormalization(momentum=momentum))
        D.add(LeakyReLU(alpha))
        D.add(Dropout(dropout))
        # state size 6: ...
        D.add(Conv2D(ndf * 8, kernel_size=3, strides=1, padding='same'))
        D.add(BatchNormalization(momentum=momentum))
        D.add(LeakyReLU(alpha))
        D.add(Dropout(dropout))
        # state size 7: ...
        D.add(Conv2D(1, kernel_size=3, strides=1, padding='same'))
        # single output layer
        D.add(Flatten())
        D.add(Dense(1, activation='sigmoid'))
        #D.summary()
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
        # input size 1: 256 x 256 x 3
        input_shape = (row, col, 3)
        input = Input(input_shape)
        X = Conv2D(ngf, kernel_size=7, strides=1, padding='same', input_shape=input_shape, name='generator_1', data_format='channels_last')(input)
        X = BatchNormalization(momentum=momentum)(X)
        X = LeakyReLU(alpha)(X)
        X = Dropout(dropout)(X)
        # state size 2: 256 x 256 x ngf
        # down convolution
        X = Conv2D(ngf*2, kernel_size=3, strides=2, padding='same')(X)
        X = Conv2D(ngf*2, kernel_size=3, strides=1, padding='same')(X)
        X = BatchNormalization(momentum=momentum)(X)
        X = LeakyReLU(alpha)(X)
        X = Dropout(dropout)(X)
        # state size 3: 128 x 128 x ngf*2
        X = Conv2D(ngf*4, kernel_size=3, strides=2, padding='same')(X)
        X = Conv2D(ngf*4, kernel_size=3, strides=1, padding='same')(X)
        X = BatchNormalization(momentum=momentum)(X)
        X = LeakyReLU(alpha)(X)
        X = Dropout(dropout)(X)
        # state size 4: 64 x 64 x ngf*4
        # resnet blocks convolution
        for i in range(nblock):
            X = self.resnet_block(X)
            X = Dropout(dropout)(X)
        # up convolution
        # state size 1 (from resblocks): 64 x 64 x ngf*4
        X = Conv2DTranspose(ngf*2, kernel_size=3, strides=2, padding='same')(X)
        X = Conv2DTranspose(ngf*2, kernel_size=3, strides=1, padding='same')(X)
        X = BatchNormalization(momentum=momentum)(X)
        X = LeakyReLU(alpha)(X)
        X = Dropout(dropout)(X)
        # state size 2: 128 x 128 x ngf*2
        X = Conv2DTranspose(ngf, kernel_size=3, strides=2, padding='same')(X)
        X = Conv2DTranspose(ngf, kernel_size=3, strides=1, padding='same')(X)
        X = BatchNormalization(momentum=momentum)(X)
        X = LeakyReLU(alpha)(X)
        X = Dropout(dropout)(X)
        # state size 3: 256 x 256 x ngf
        X = Conv2D(3, kernel_size=7, strides=1, padding='same')(X)
        # output state: 256 x 256 x 3
        G = Model(inputs=input, outputs=X)
        #G.summary()
        self.G = G
        return self.G

    def resnet_block(self, X):
        nf = self.args['ngfeature']*4   # 256
        alpha = 0.0
        momentum = 0.1
        # ====== resnet block ======
        last = X
        X = Conv2D(nf, kernel_size=3, strides=1, padding='same')(X)
        X = BatchNormalization(momentum=momentum)(X)
        X = LeakyReLU(alpha)(X)
        X = Conv2D(nf, kernel_size=3, strides=1, padding='same')(X)
        X = BatchNormalization(momentum=momentum)(X)
        X = Add()([last, X])
        return X

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

    # ===== util methods ======
    def adversarial_loss(self):

        def loss(y_true, y_pred):
            return K.mean()

        return loss

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
        save_param = self.args['save_param']
        for epoch in range(nepoch):
            log = 'epoch: %d / %d' % (epoch, nepoch)
            print(log)
            for batch in range(nbatch):
                log = 'batch %d / %d' % (batch, nbatch)
                print(log)
                # ====== train discriminator ======
                real, noise = self.get_train_batch()
                real_size, noise_size = len(real), len(noise)
                fake = self.generator.predict(noise)
                x = np.concatenate((real, fake))
                y = np.concatenate((np.ones([real_size, 1]), np.zeros([noise_size, 1])))
                d_loss = self.discriminator.train_on_batch(x,y)
                # ====== train adversarial ======
                y = np.ones([noise_size, 1])
                a_loss = self.adversarial.train_on_batch(noise, y)
                # ====== log message ======
                log = 'discriminator: [loss: %d, acc: %d]\n' % (d_loss[0], d_loss[1])
                log += 'adversarial: [loss: %d, acc: %d]' % (a_loss[0], a_loss[1])
                print(log)
        log = '------ finish training ------'
        print(log)
        log = '------ saving model ------'
        print(log)
        # ====== save model ======
        self.save(save_param, self.generator)
        log = '------ parameters saved ------'
        print(log)

    def test(self, plot_sample=False):
        log = '------ start testing ------'
        print(log)
        test_batch = self.get_test()
        print(test_batch.shape)
        result = self.generator.predict(test_batch)
        if plot_sample:
            image = result[0]
            self.plot_image(image)
        log = '------ finish testing ------'
        print(log)
        return result

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

        return cartoon_batch[0], photo_batch[0]

    '''
    def get_train_batch(self):
        ncbatch = len(self.cartoon_batch)
        npbatch = len(self.photo_batch)
        cindex = np.random.randint(0, ncbatch)
        pindex = np.random.randint(0, npbatch)
        return self.cartoon_batch[cindex], self.photo_batch[pindex]
    '''

    def get_test(self):
        test_dir = self.args['test_dir']
        row = self.args['row']
        col = self.args['col']
        zero_dir = os.path.join(test_dir, '0')
        batch_size = len([f for f in os.listdir(zero_dir) if os.path.isfile(os.path.join(zero_dir, f))])
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
        return test[0]

    def plot_image(self, image):
        plt.imshow(image)
        plt.show()

    def save(self, fname, model):
        model.save_weights(fname)

def arguments():
    args = {
    # ====== data level ======
        'cartoon_dir' : 'data/train/cartoon',
        'photo_dir' : 'data/train/photo',
        'test_dir' : 'data/test',
        'save_param' : 'gen_param.h5',
    # ====== train level ======
        'nepoch_pretrain' : 10,
        'nepoch' : 50,
        'nbatch' : 100,
        'batch_size' : 4,
    # ====== model level ======
        'row' : 256,
        'col' : 256,
        'ndfeature' : 8,
        'ngfeature' : 16,
        'nblock' : 8,
        'nblock' : 8,
        'lrD' : 1e-5,
        'lrG' : 1e-5
    }
    return args

# ====== main method ======
args = arguments()
model = model(args)
#model.train()
model.generator.load_weights('gen_param.h5')
result = model.test()
plt.imshow(result[0])
plt.show()
'''
np.save('result.npy', result)
result = np.load('result.npy')
'''
