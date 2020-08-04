import warnings
import wfdb
import os
import numpy as np
import operator
from utils import *
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
warnings.filterwarnings("ignore")
data_path='D:/python/bwl res_net/MIT-BIH'
DS1=[101,106,108,109,112,114,115,116,118,119,122,124,201,203,205,207,208,209,215,220,223,230]
DS2=[100,103,105,111,113,117,121,123,200,202,210,212,213,214,219,221,222,228,231,232,233,234]
data = {'train': DS1,
        'test': DS2
        }
length=1800
target_class = ['train', 'test']
sign_class = ['N', 'S', 'V','F']
#MITBIH_classes = ['N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F','/','Q','f']
def MaxMinNormalization(x):
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x
def Z_ScoreNormalization(x):
    x = (x - np.average(x)) / np.std(x)
    return x
def creat_data():
    for i in range(len(target_class)):
        s=data[target_class[i]]
        beat=[]
        label_R_peak=np.empty((0,length))
        label_class=np.empty((0,length,len(sign_class)))
        for k in range(len(s)):
            start_time = 0
            end_time = start_time + length
            print(data_path + '/' + str(s[k]) + target_class[i])
            record = wfdb.rdrecord(data_path + '/' + str(s[k]), sampfrom=0, channel_names=['MLII'])
            sigal = record.p_signal
            sigal = Z_ScoreNormalization(sigal)
            annotation = wfdb.rdann(data_path + '/' + str(s[k]), 'atr')
            while end_time <= sigal.shape[0]:
                sign = sigal[start_time:end_time]
                ab_normal=0
                inivit_r=np.zeros((1,length))
                inivit_class = np.ones((1,length,len(sign_class)))*(-1)
                for j in range(annotation.ann_len - 1):
                    cla=-1
                    if annotation.sample[j] >= start_time and annotation.sample[j] < end_time:
                        inivit_r[0][annotation.sample[j]-start_time]=1
                        if annotation.symbol[j] == 'N' or annotation.symbol[j] == 'L' or annotation.symbol[j] == 'R' or \
                                annotation.symbol[j] == 'e' or annotation.symbol[j] == 'j':
                            cla=0
                        elif annotation.symbol[j] == 'A' or annotation.symbol[j] == 'a' or annotation.symbol[
                            j] == 'J' or \
                                annotation.symbol[j] == 'S':
                            cla=1
                        elif annotation.symbol[j] == 'V' or annotation.symbol[j] == 'E':
                            cla = 2
                        elif annotation.symbol[j] == 'F':
                            cla = 3
                        else:
                            ab_normal=1
                        for index in range(len(sign_class)):
                            if index==cla:
                                inivit_class[0][annotation.sample[j] - start_time][index] = 1
                            else:
                                inivit_class[0][annotation.sample[j] - start_time][index] = 0
                if ab_normal==0:
                    beat.append(sign)
                    label_R_peak = np.concatenate((label_R_peak, inivit_r), axis=0)
                    label_class = np.concatenate((label_class, inivit_class), axis=0)
                start_time = end_time
                end_time = start_time + length
        beat = np.asarray(beat, dtype=np.float32)  # 将训练的图像数据原来是list现在变成np.array格式
        label_R_peak = np.asarray(label_R_peak, dtype=int)
        label_class = np.asarray(label_class, dtype=int)
        print(beat.shape,label_R_peak.shape,label_class.shape)
        np.save('D:/python/multiple_label_new/npy/beat_'+target_class[i]+'_seg.npy', beat)
        np.save('D:/python/multiple_label_new/npy/label_R_peak_'+target_class[i]+'_seg.npy', label_R_peak)
        np.save('D:/python/multiple_label_new/npy/label_class_'+target_class[i]+'_seg.npy', label_class)
creat_data()
