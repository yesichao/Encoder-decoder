from utils import *
import os
target_class = ['train', 'test']
sig_class=['N','S','V','F']
if os.path.isfile('D:/python/multiple_label_new/npy/true_label_class_train_seg.npy') \
            and os.path.isfile('D:/python/multiple_label_new/npy/true_label_class_test_seg.npy'):
    print('loading...')
    true_l_class_tr = np.load('D:/python/multiple_label_new/npy/true_label_class_train_seg.npy')
    true_l_class_te = np.load('D:/python/multiple_label_new/npy/true_label_class_test_seg.npy')
else:
    for i in range(len(target_class)):
        label=np.load('D:/python/multiple_label_new/npy/label_class_'+target_class[i]+'_seg.npy')
        print(label.shape)
        true_l_R=[]
        true_l_class = []
        for j in range(label.shape[0]):
            s=np.where(label[j]==1)
            true_l_R.append(s[0])
            true_l_class.append(s[1])
        print(len(true_l_R))
        print(len(true_l_class))
        np.save('D:/python/multiple_label_new/npy/true_label_R_'+target_class[i]+'_seg.npy',true_l_R)
        np.save('D:/python/multiple_label_new/npy/true_label_class_'+target_class[i]+'_seg.npy',true_l_class)
for i in range(len(target_class)):
    true_l_class = np.load('D:/python/multiple_label_new/npy/true_label_class_'+target_class[i]+'_seg.npy')
    global_class = np.empty((0, 4))
    for m in range(true_l_class.shape[0]):
        mid_global_class=np.zeros((1,4))
        for j in range(len(true_l_class[m])):
            for s in range(4):
                if true_l_class[m][j]==s:
                    mid_global_class[0][s]=1
        global_class=np.concatenate((global_class, mid_global_class), axis=0)
    print(global_class.shape)
    np.save('D:/python/multiple_label_new/npy/global_class_'+target_class[i]+'_seg.npy',global_class)