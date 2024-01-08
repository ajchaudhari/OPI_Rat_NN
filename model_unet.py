#################################################################################

#################################################################################

import tensorflow as tf

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.layers import Input, MaxPooling2D, UpSampling2D, Conv2D
from keras.layers.normalization import BatchNormalization

#### Modified 2D U-Net Model ####

dropout = 0.2 #dropout rate

filterSize = 3 #convolution filter size

actF = tf.keras.layers.LeakyReLU(alpha=0.1) #activiation function

hn = 'random_uniform' #kernel initializer

def unet(input_size, numClasses):
    
    inputs = Input(input_size)

    conv1 = Conv2D(64, filterSize, activation = actF, padding = 'same', kernel_initializer = hn)(inputs)
    conv1 = Conv2D(64, filterSize, activation = actF, padding = 'same', kernel_initializer = hn)(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, filterSize, activation = actF, padding = 'same', kernel_initializer = hn)(pool1)
    conv2 = Conv2D(128, filterSize, activation = actF, padding = 'same', kernel_initializer = hn)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, filterSize, activation = actF, padding = 'same', kernel_initializer = hn)(pool2)
    conv3 = Conv2D(256, filterSize, activation = actF, padding = 'same', kernel_initializer = hn)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, filterSize, activation = actF, padding = 'same', kernel_initializer = hn)(pool3)
    conv4 = Conv2D(512, filterSize, activation = actF, padding = 'same', kernel_initializer = hn)(conv4)
    drop4 = Dropout(dropout)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, filterSize, activation = actF, padding = 'same', kernel_initializer = hn)(pool4)
    conv5 = Conv2D(1024, filterSize, activation = actF, padding = 'same', kernel_initializer = hn)(conv5)
    drop5 = Dropout(dropout)(conv5)

    up6 = Conv2D(512, 2, activation = actF, padding = 'same', kernel_initializer = hn)(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, filterSize, activation = actF, padding = 'same', kernel_initializer = hn)(merge6)
    conv6 = Conv2D(512, filterSize, activation = actF, padding = 'same', kernel_initializer = hn)(conv6)

    up7 = Conv2D(256, 2, activation = actF, padding = 'same', kernel_initializer = hn)(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, filterSize, activation = actF, padding = 'same', kernel_initializer = hn)(merge7)
    conv7 = Conv2D(256, filterSize, activation = actF, padding = 'same', kernel_initializer = hn)(conv7)

    up8 = Conv2D(128, 2, activation = actF, padding = 'same', kernel_initializer = hn)(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, filterSize, activation = actF, padding = 'same', kernel_initializer = hn)(merge8)
    conv8 = Conv2D(128, filterSize, activation = actF, padding = 'same', kernel_initializer = hn)(conv8)

    up9 = Conv2D(64, 2, activation = actF, padding = 'same', kernel_initializer = hn)(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, filterSize, activation = actF, padding = 'same', kernel_initializer = hn)(merge9)
    conv9 = Conv2D(64, filterSize, activation = actF, padding = 'same', kernel_initializer = hn)(conv9)
    
    # Softmax
    conv10 = Conv2D(numClasses, (1,1), activation = 'softmax')(conv9)

    model = Model(inputs = inputs, outputs = conv10)

    return model 
