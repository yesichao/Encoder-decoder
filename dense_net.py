from keras.layers import Add
from keras.layers import MaxPooling1D
from keras.models import Model
from keras.models import Input
from keras.layers import Conv1D,Dropout,concatenate
from keras.layers import BatchNormalization
from keras.layers.convolutional import UpSampling1D
from keras.layers.core import Activation
from keras.layers import GlobalAveragePooling1D
from keras.layers.core import Dense
from keras import backend as K
import numpy as np
def DenseLayer(x, nb_filter, bn_size=4, drop_rate=0.2):
    # Bottleneck layers
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(bn_size*nb_filter,1, strides=1, padding='same', kernel_initializer='he_normal')(x)
    # Composite function
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(nb_filter, 32, strides=1, padding='same', kernel_initializer='he_normal')(x)
    if drop_rate: x = Dropout(drop_rate)(x)
    return x
def DenseBlock(x, nb_layers, growth_rate, drop_rate=0.2):
    for ii in range(nb_layers):
        conv = DenseLayer(x, nb_filter=growth_rate, drop_rate=drop_rate)
        x = concatenate([x, conv], axis=-1)
    return x
def TransitionLayer(x, compression=0.5):
    nb_filter = int(x.shape.as_list()[-1]*compression)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(32, 1, strides=1, padding='same', kernel_initializer='he_normal')(x)
    #x = MaxPooling1D(pool_size=2, strides=2)(x)
    return x
def decode(pre_encode,con_encode,n_fil,s_fil):
    up1 = Conv1D(n_fil, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling1D(size=2)(pre_encode))
    merge = concatenate([up1, con_encode], axis=-1)
    conv = Conv1D(n_fil,s_fil, activation='relu', padding='same', kernel_initializer='he_normal')(merge)
    conv = Conv1D(n_fil,s_fil, activation='relu', padding='same', kernel_initializer='he_normal')(conv)
    return conv
def Dense_Net(growth_rate=16):
    input = Input(shape=[1800,1],
                   dtype='float32',
                   name='inputs')
    encode_in = Conv1D(growth_rate * 2, 32, strides=1, padding='same', kernel_initializer='he_normal')(input)
    encode_1 = DenseBlock(encode_in, 5, growth_rate, drop_rate=0.2)
    encode_1 = TransitionLayer(encode_1)
    encode_max_1 = MaxPooling1D(pool_size=2, strides=2)(encode_1)
    encode_2 = DenseBlock(encode_max_1, 5, growth_rate, drop_rate=0.2)
    encode_2 = TransitionLayer(encode_2)
    encode_max_2 = MaxPooling1D(pool_size=2, strides=2)(encode_2)
    encode_3 = DenseBlock(encode_max_2, 5, growth_rate, drop_rate=0.2)
    encode_3 = TransitionLayer(encode_3)
    encode_max_3 = MaxPooling1D(pool_size=2, strides=2)(encode_3)
    encode_4 = DenseBlock(encode_max_3, 5, growth_rate, drop_rate=0.2)
    encode_4 = TransitionLayer(encode_4)
    dilation_out1 = Conv1D(32,
                           kernel_size=8,
                           dilation_rate=1,
                           padding='same')(encode_4)
    dilation_out2 = Conv1D(32, kernel_size=8, dilation_rate=2, padding='same')(dilation_out1)
    dilation_out3 = Conv1D(32, kernel_size=8, dilation_rate=4, padding='same')(dilation_out2)
    dilation_out4 = Conv1D(32, kernel_size=8, dilation_rate=8, padding='same')(dilation_out3)
    dilation_out = Add()([encode_4, dilation_out1, dilation_out2, dilation_out3, dilation_out4])
    decode_1=decode(dilation_out,encode_3,32,32)
    decode_2 = decode(decode_1, encode_2, 32, 32)
    decode_3 = decode(decode_2, encode_1, 32, 32)
    #r_layer = Conv1D(8, 32, strides=1, padding='same', kernel_initializer='he_normal')(decode_3)
    r_layer=Conv1D(1,1,strides=1, padding='same',kernel_initializer='he_normal')(decode_3)
    r_output = Activation('sigmoid', name='r_output')(r_layer)
    model = Model(inputs=[input], outputs=[r_output])
    model.summary()
    return model