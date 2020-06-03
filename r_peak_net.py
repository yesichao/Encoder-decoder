from keras.layers import Add
from keras.layers import MaxPooling1D
from keras.models import Model
from keras.models import Input
from keras.layers import Conv1D,Flatten
from keras.layers import BatchNormalization
from keras.layers.convolutional import UpSampling1D
from keras.layers.core import Dense, Activation
from keras.layers import Dropout
from keras.layers import GlobalAveragePooling1D
def identity_block(X,filter,sizes,lam):
    S1, S2 = sizes
    shortcut = X
    layer=Activation('relu')(X)
    layer = Conv1D(filter, S1, strides=1, padding='same',kernel_initializer='he_normal')(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    #layer = Dropout(rate=0.2)(layer)
    layer = Conv1D(filter, S2, strides=1, padding='same',kernel_initializer='he_normal')(layer)
    layer = BatchNormalization()(layer)
    if lam==1:
        shortcut=Conv1D(filter,1, strides=1, padding='same',kernel_initializer='he_normal')(shortcut)
    layer = Add()([shortcut, layer])
    layer = Activation('relu')(layer)

    return layer
def upconv_bn_relu(X,out_channel,size,stride):
    x=transposeconv(X,out_channel,size,stride)
    x = BatchNormalization()(x)
    x =  Activation('relu')(x)
    return x
def decoder_block(x, num_channel_m, num_channel_n, kernel_size, stride=1):
    x = upconv_bn_relu(x, num_channel_m // 4, 1, 1,)
    x = upconv_bn_relu(x, num_channel_m // 4,kernel_size, stride)
    x = upconv_bn_relu(x, num_channel_n, 1, 1)
    return x
def transposeconv(layer,filters,
                               size,
                               stride):
    x=UpSampling1D(size=stride)(layer)
    x=Conv1D(filters,size,strides=1, padding='same',kernel_initializer='he_normal')(x)
    return x
def Net():
    inputs =  Input(shape=[1800,1],
                   dtype='float32',
                   name='inputs')
    filters = [16,32,64,128]
    filters_m = [16,32,64,128][::-1]
    filters_n = [16,16,32,64][::-1]
    layer=Conv1D(filters[0],32,strides=1, padding='same',kernel_initializer='he_normal')(inputs)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    layer=MaxPooling1D(pool_size=2, strides=2)(layer)
    for i in range(3):
        if i==0:
            lam=1
        else:
            lam=0
        layer=identity_block(layer,filters[0],[32,32],lam)
    encoder_block_1 = layer
    layer=MaxPooling1D(pool_size=2, strides=2)(layer)
    for i in range(4):
        if i==0:
            lam=1
        else:
            lam=0
        layer=identity_block(layer,filters[1],[32,32],lam)
    encoder_block_2 = layer
    layer = MaxPooling1D(pool_size=2, strides=2)(layer)
    for i in range(6):
        if i==0:
            lam=1
        else:
            lam=0
        layer=identity_block(layer,filters[2],[32,32],lam)
    encoder_block_3 = layer
    layer = MaxPooling1D(pool_size=2, strides=2)(layer)
    for i in range(3):
        if i==0:
            lam=1
        else:
            lam=0
        layer = identity_block(layer, filters[3], [32,32],lam)
    dilation_out1=Conv1D(filters[3],
                         kernel_size=3,
                         dilation_rate=1,
                         padding='same')(layer)
    dilation_out2 = Conv1D(filters[3], kernel_size=32, dilation_rate=2, padding='same')(dilation_out1)
    dilation_out3 = Conv1D(filters[3], kernel_size=32, dilation_rate=4, padding='same')(dilation_out2)
    dilation_out4 = Conv1D(filters[3], kernel_size=32, dilation_rate=8, padding='same')(dilation_out3)
    layer=Add()([layer,dilation_out1,dilation_out2,dilation_out3,dilation_out4])
    layer=decoder_block(layer, filters_m[0], filters_n[0], 32,2)
    layer=Add()([layer,encoder_block_3])
    layer = decoder_block(layer, filters_m[1], filters_n[1], 32, 2)
    layer = Add()([layer, encoder_block_2])
    layer = decoder_block(layer, filters_m[2], filters_n[2], 32, 2)
    layer = Add()([layer, encoder_block_1])
    layer = decoder_block(layer, filters_m[3], filters_n[3], 32, 2)
    layer=transposeconv(layer,8,32,1)
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    r_layer=Conv1D(6,32,strides=1, padding='same',kernel_initializer='he_normal')(layer)
    r_layer=Conv1D(1,1,strides=1, padding='same',kernel_initializer='he_normal')(r_layer)
    r_output = Activation('sigmoid', name='r_output')(r_layer)
    model = Model(inputs=[inputs], outputs=[r_output])
    model.summary()
    return model