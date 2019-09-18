# a trained model that distinguished whether there is a character figure
# that occupies a large area in the given picture

import numpy as np
from PIL import Image

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D                    #CNN layers
from keras.layers import Dense, Activation, Dropout, Flatten     # common layers
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K

import os
import shutil
import time

#####################
# 1. train model    #
#####################

def build_model():
    model = Sequential()
    # CNN layer 1
    if K.image_data_format() == 'channels_first':
        shape = (3, w, h)
    else:
        shape = (w, h, 3)
    model.add(Conv2D(32, (3, 3), input_shape=shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # CNN layer 2
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # CNN layer 3
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # linear layer: binary classification
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    # dropout
    model.add(Dropout(0.5))
    # output layer
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    # model compile
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    # return model
    return model

def train(weight_path, w, h, batch_size, train_size, model):
    train_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.1,
        zoom_range = 0.1,
        horizontal_flip = True)

    train_gen = train_datagen.flow_from_directory(
        'data/train',
        target_size = (w, h),
        batch_size = batch_size,
        class_mode = 'binary'
        )
    
    valid_datagen = ImageDataGenerator(rescale = 1./255)
    
    valid_gen = valid_datagen.flow_from_directory(
        'data/validation',
        target_size = (w, h),
        batch_size = 10,
        class_mode = 'binary'
        )

    model.fit_generator(
        train_gen,
        steps_per_epoch = train_size // batch_size,
        validation_data = valid_gen,
        validation_steps = 5,
        epochs = 40
        )

    model.save_weights(weight_path)

# raw picture size: w, h = 1290, 692
w, h = 430, 230
batch_size = 16
'''
fig_size = len(next(os.walk('\\Users\calin\Desktop\AutoComics\Figure-Recognition\data\\train\figure'))[2])
nonfig_size = len(next(os.walk('\\Users\calin\Desktop\AutoComics\Figure-Recognition\data\\train\nonfigure'))[2])
train_size = (fig_size + nonfig_size)*5
'''
train_size = 436 * 5
model = build_model()

##train('figure_recog_v3.h5', w, h, batch_size, train_size, model)

################################
# 2. apply figure recognition  #
################################

def apply_recognition(sample_dir, output_dir):
    
    dir_list = os.listdir(sample_dir)
    
    for f in dir_list:
        if os.path.isfile(output_dir+'/figure/'+f) or os.path.isfile(output_dir+'/nonfigure/'+f):
    ##        print(f+' already exists')
            continue
        shutil.copy(sample_dir+'/'+f, 'temp/figure')
        
        test_datagen = ImageDataGenerator(rescale = 1./255)
        test_gen = test_datagen.flow_from_directory(
            'temp',
            target_size = (w, h),
            batch_size = 1,
            class_mode = 'binary')

        prediction = model.predict_generator(
            test_gen,
            steps = 1)
        prediction = np.round(prediction[0][0])
##        print(prediction)
        
        # label 1 is non-figure label, 0 is figure label
        if prediction == 1:
            output = output_dir + '/nonfigure'
        else:
            output = output_dir + '/figure'
        # exception
        try:
            shutil.move('temp/figure/'+f, output)
            time.sleep(0.5)
        except:
            time.sleep(5)
            os.remove('temp/figure/'+f)

model.load_weights('boolFigure_v3.h5')

sample_dir = '\\Users\calin\Desktop\AutoComics\data\\Totoro_smooth_pix3'
output_dir = '\\Users\calin\Desktop\AutoComics\data' # a directory that contains 2 subdirectories: figure, nonfigure

apply_recognition(sample_dir, output_dir)

