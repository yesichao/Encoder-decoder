from keras.optimizers import SGD,Adam
from keras.models import Model
from keras.models import Input
from keras.layers import Conv1D
from keras.layers.core import Dense, Activation
from keras.layers import Dropout,Flatten
def lr_schedule(epoch):
    lr = 0.0001*((1-0.0001)**epoch)
    print('Learning rate: ', lr)
    return lr

shape_1=[0,241,121,61,31,61,121,241]
shape_3=[0,32,32,32,32,32,32,32]
shape_3=[0,56,68,74,77,32,32,32]
def Net(map_num):
    inputs =  Input(shape=[shape_1[map_num],shape_3[map_num]],
                   dtype='float32',
                   name='inputs')
    layer=Conv1D(32, 8, strides=1, padding='same',activation='relu')(inputs)
    layer=Flatten()(layer)
    layer = Dense(128)(layer)
    output=Dense(4,activation='softmax')(layer)
    model = Model(inputs=[inputs], outputs=[output])
    optimizer =Adam(lr=lr_schedule(0))
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['categorical_accuracy'])
    model.summary()
    return model