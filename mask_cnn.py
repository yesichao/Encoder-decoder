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
from keras.models import load_model
import numpy as np
def Net():
    #model_name = 'D:/python/multiple_label_new/model_t/class_res_' + str(178) + '_net.h5'
    model_name = 'C:/Users/叶思超/Desktop/r峰结果/新建文件夹/1/class_res_' + str(129) + '_net.h5'
    #model_name = 'C:/Users/叶思超/Desktop/r峰结果/model_t/class_res_' + str(90) + '_net.h5'
    r_model = load_model(model_name)
    '''feature_map_1 = r_model.get_layer('conv1d_12').output  # 1800
feature_map_2 = r_model.get_layer('conv1d_23').output  # 900
feature_map_3=r_model.get_layer('conv1d_34').output#450
feature_map_4 = r_model.get_layer('add_1').output  # 225
feature_map_5 = r_model.get_layer('conv1d_50').output#450
feature_map_6 = r_model.get_layer('conv1d_53').output# 900
feature_map_7 = r_model.get_layer('conv1d_56').output# 1800'''
    feature_map_1 = r_model.get_layer('conv1d_71').output  # 1800
    feature_map_2 = r_model.get_layer('conv1d_82').output  # 900
    feature_map_3=r_model.get_layer('conv1d_93').output#450
    feature_map_4 = r_model.get_layer('add_2').output  # 225
    feature_map_5 = r_model.get_layer('conv1d_109').output#450
    feature_map_6 = r_model.get_layer('conv1d_112').output# 900
    feature_map_7 = r_model.get_layer('conv1d_115').output# 1800
    r_output = r_model.get_layer('r_output').output
    model = Model(inputs=r_model.input,
                  outputs = [feature_map_1,feature_map_2,feature_map_3,feature_map_4,
                             feature_map_5,feature_map_6,feature_map_7,r_output])
    model.summary()
    return model
