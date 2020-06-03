import numpy as np
import math
classes=['tr','te']
classes_1=['train','test']
def label_val(R_loc,true_l_R,true_l_class):
    cm=np.zeros((4))
    fin_r_loc=[]
    fin_l_class=[]
    for j in range(true_l_class.shape[0]):
        class_lab = true_l_class[j][:].tolist()
        r_lab = true_l_R[j][:].tolist()
        r_loc = R_loc[j][:]
        fin_mid_r_loc=[]
        fin_mid_l_class=[]
        num=0
        # print(class_lab,class_pre,r_loc,r_lab)
        for s in range(len(true_l_R[j])):
            for i in range(len(r_loc)):
                if math.fabs(r_loc[i] - true_l_R[j][s]) < 0.15 * 360:
                    fin_mid_r_loc.append(r_loc[i])
                    fin_mid_l_class.append(class_lab[s])
                    class_lab[s] = -1
                    del r_loc[i]
                    break
        for m in range(len(class_lab)):
            if class_lab[m]==-1:
                num=num+1
        for m in range(num):
            class_lab.remove(-1)
        for m in range(len(class_lab)):
            cm[class_lab[m]] = cm[class_lab[m]] + 1
        fin_r_loc.append(fin_mid_r_loc)
        fin_l_class.append(fin_mid_l_class)
    return fin_l_class,fin_r_loc,cm
sum_mid=[62595,26890]
shape_3=[0,32,32,32,32,32,32,32]
narrow_rate=[0,1,2,4,8,4,2,1]
for j in range(1,8):
    for i in range(len(classes)):
        fin_feature=[]
        feature_map=np.load('D:/python/multiple_label_new/feature_map/feature_map_'+str(j)+'_'+classes[i]+'.npy')
        true_l_R = np.load('D:/python/multiple_label_new/npy_nds/true_label_R_'+classes_1[i]+'_seg.npy')
        true_l_class= np.load('D:/python/multiple_label_new/npy_nds/true_label_class_'+classes_1[i]+'_seg.npy')
        R_loc=np.load('D:/python/multiple_label_new/feature_map/r_loc_'+classes[i]+'.npy')
        print(feature_map.shape)
        fin_l_class, fin_r_loc, cm=label_val(R_loc, true_l_R, true_l_class)
        print(cm)
        print(fin_r_loc)
        print(fin_l_class)
        feature_sum=np.empty((sum_mid[i],int(240/narrow_rate[j])+1 ,shape_3[j]))#随图的位置有变化，懒得写集合,还是写了
        label_sum=[]
        sum_num = 0
        for m in range(feature_map.shape[0]):
            for n in range(len(fin_r_loc[m])):
                if int(fin_r_loc[m][n]/narrow_rate[j])<int(96/narrow_rate[j]):#随图的位置有变化
                    fin_feature=feature_map[m][0:int(fin_r_loc[m][n]/narrow_rate[j])+int(144/narrow_rate[j])+1][:]
                    fin_feature = np.asarray(fin_feature, dtype=np.float32)
                    b=np.zeros((int(96/narrow_rate[j])-int(fin_r_loc[m][n]/narrow_rate[j]),shape_3[j]))
                    fin_feature=np.concatenate((b,fin_feature),axis=0)
                elif int(fin_r_loc[m][n]/narrow_rate[j])+int(144/narrow_rate[j])+1>int(1800/narrow_rate[j]):
                    fin_feature = feature_map[m][int(fin_r_loc[m][n] / narrow_rate[j]) - int(96/narrow_rate[j]):int(1800/narrow_rate[j])][:]
                    fin_feature = np.asarray(fin_feature, dtype=np.float32)
                    b = np.zeros((int(fin_r_loc[m][n] / narrow_rate[j])+int(144/narrow_rate[j])+1-int(1800/narrow_rate[j]), shape_3[j]))
                    fin_feature = np.concatenate(( fin_feature,b), axis=0)
                else:
                    fin_feature = feature_map[m][int(fin_r_loc[m][n] / narrow_rate[j]) -int(96/narrow_rate[j]):int(fin_r_loc[m][n] / narrow_rate[j])+int(144/narrow_rate[j])+1][:]
                    fin_feature = np.asarray(fin_feature, dtype=np.float32)
                #fin_feature=np.expand_dims(fin_feature, axis=0)
                feature_sum[sum_num,:,:]=fin_feature
                sum_num=sum_num+1
                label_sum.append(fin_l_class[m][n])
        label_sum = np.asarray(label_sum, dtype=np.float32)
        print(feature_sum.shape)
        print(label_sum.shape)
        np.save('D:/python/multiple_label_new/fin_feature/fin_feature_'+str(j)+'_'+classes[i]+'.npy',feature_sum)
        np.save('D:/python/multiple_label_new/fin_feature/fin_label_' + classes[i] + '.npy',
                label_sum)