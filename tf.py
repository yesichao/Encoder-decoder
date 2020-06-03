import numpy as np
import math
from keras.models import load_model
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #二维图像必须用到的库
import numpy as np
sig_class = ['N', 'S', 'V', 'F']
true_l_R_tr = np.load('D:/python/multiple_label_new/npy_nds/2/true_label_R_train_seg.npy')
true_l_R_te = np.load('D:/python/multiple_label_new/npy_nds/2/true_label_R_test_seg.npy')
true_l_class_tr = np.load('D:/python/multiple_label_new/npy_nds/2/true_label_class_train_seg.npy')
true_l_class_te = np.load('D:/python/multiple_label_new/npy_nds/2/true_label_class_test_seg.npy')
train_sig = np.load('D:/python/multiple_label_new/npy_nds/2/beat_train_seg.npy')
test_sig = np.load('D:/python/multiple_label_new/npy_nds/2/beat_test_seg.npy')
label_R_tr = np.load('D:/python/multiple_label_new/npy_nds/2/label_R_peak_train_seg.npy')
label_R_te = np.load('D:/python/multiple_label_new/npy_nds/2/label_R_peak_test_seg.npy')
label_class_tr = np.load('D:/python/multiple_label_new/npy_nds/2/label_class_train_seg.npy')
label_class_te = np.load('D:/python/multiple_label_new/npy_nds/2/label_class_test_seg.npy')
global_class_tr = np.load('D:/python/multiple_label_new/npy_nds/2/global_class_train_seg.npy')
global_class_te = np.load('D:/python/multiple_label_new/npy_nds/2/global_class_test_seg.npy')
R_loc_te=np.load('D:/python/multiple_label_new/feature_map/r_loc_te.npy')
feature_sum_te=np.load('D:/python/multiple_label_new/fin_feature/fin_feature_3_te.npy')

#feature_sum_te=np.expand_dims(feature_sum_te, axis=3)
label_sum_te=np.load('D:/python/multiple_label_new/fin_feature/fin_label_te.npy')
print(label_sum_te[115])
label_sum_te=to_categorical(label_sum_te, num_classes=len(sig_class))
model_name = 'D:/python/multiple_label_new/fin_model/my_model_1.hdf5'

'''fig = plt.figure()
ax = Axes3D(fig)
x1=np.arange(0,32,1)
x2=np.arange(0,61,1)

x1, x2 = np.meshgrid(x1, x2)#网格的创建，这个是关键
plt.xlabel('x1')
plt.ylabel('x2')
ax.plot_surface(x1, x2, feature_sum_te[1], rstride=1, cstride=1, cmap='rainbow')
plt.show()'''
i=115

plt.imshow(feature_sum_te[i].T)
#plt.savefig('infor_acc.eps', dpi=300, format='eps')
plt.show()
text_fea_1 = feature_sum_te[i].T
print(text_fea_1[:,20:].shape)
text_fea = np.expand_dims(feature_sum_te[i], axis=0)

print(feature_sum_te[i].shape)
model = load_model(model_name)
pred_vt = model.predict(text_fea, verbose=1)
pred_v = np.argmax(pred_vt, axis=1)
print(pred_v)
cm=np.zeros((32,20))

text_fea=np.concatenate((cm,text_fea_1[:,20:]),axis=1)
plt.imshow(text_fea)
plt.savefig('infor_acc_1.eps', dpi=300, format='eps')
plt.show()
text_fea=np.expand_dims(text_fea.T, axis=0)
pred_vt = model.predict(text_fea, verbose=1)
pred_v = np.argmax(pred_vt, axis=1)
print(pred_v)
def label_val1(R_loc,true_l_R,true_l_class,stan):
    FP=0
    FN=0
    TN=0
    for j in range(true_l_class.shape[0]):
        r_loc = R_loc[j][:]
        FP=FP+len(r_loc )
        FN=FN+len(true_l_R_te[j])
        for s in range(len(true_l_R[j])):
            for i in range(len(r_loc)):
                if math.fabs(r_loc[i] - true_l_R[j][s]) <= stan * 360:
                    TN = TN + 1
                    del r_loc[i]
                    break
    TP = true_l_class.shape[0] * 1800 - FN
    FN = FN - TN
    FP = FP - TN
    TP = TP - FP
    cm = np.zeros((2, 2))
    cm[0][0] = TN  # 写反了，算了，懒得改  TP   FN
    cm[0][1] = FN  # FP   TN
    cm[1][1] = TP
    cm[1][0] = FP
    cm = np.asarray(cm, dtype=int)
    overall_accuracy = (TN) / (TN + FP + FN)
    return overall_accuracy
'''o_acc=[]
i=0
num=[]
while i<=0.153:
    acc=label_val1(R_loc_te,true_l_R_te,true_l_class_te,i)
    o_acc.append(acc*100)
    num.append(int(i*1000))
    i=i+0.003

np.save('num.npy',num)
num=np.load('num.npy')
o_acc=np.load('o_acc.npy')
m=[0,1,4,12,14,20,28,32]
print(num[m])
print(o_acc[m])
plt.plot(num,o_acc)
plt.xlim(0,160)
plt.ylim(87,100)
plt.xlabel('Error range/ms')
plt.ylabel('Accuracy/%')
plt.scatter(num[m],o_acc[m],c='r',s=30,marker='x')
plt.savefig('o_acc.eps', dpi=600, format='eps')
plt.show()'''
'''def label_val1(R_loc,true_l_R,true_l_class,segn):
    cm=np.zeros((4))
    fin_r_loc=[]
    fin_l_class=[]
    fase=[]
    tru=[]
    seg=[]
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
        if len(r_loc)!=0:
            fase.append(R_loc[j])
            tru.append(true_l_R[j])
            seg.append(segn[j])
        for m in range(num):
            class_lab.remove(-1)
        for m in range(len(class_lab)):
            cm[class_lab[m]] = cm[class_lab[m]] + 1
    return fase,tru,seg

def label_val(R_loc,true_l_R,true_l_class,segn):
    cm=np.zeros((4))
    fin_r_loc=[]
    fin_l_class=[]
    fase=[]
    tru=[]
    seg=[]
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
                if math.fabs(r_loc[i] - true_l_R[j][s]) < 0.08 * 360:
                    fin_mid_r_loc.append(r_loc[i])
                    fin_mid_l_class.append(class_lab[s])
                    class_lab[s] = -1
                    del r_loc[i]
                    break
        for m in range(len(class_lab)):
            if class_lab[m]==-1:
                num=num+1
        if num!=len(class_lab):
            fase.append(R_loc[j])
            tru.append(true_l_R[j])
            seg.append(segn[j])
        for m in range(num):
            class_lab.remove(-1)
        for m in range(len(class_lab)):
            cm[class_lab[m]] = cm[class_lab[m]] + 1
    return fase,tru,seg
fase,tru,seg=label_val(R_loc_te, true_l_R_te, true_l_class_te,test_sig)
print(len(tru))
for i in range(17,18):
    plt.plot(seg[i])
    plt.scatter(fase[i],seg[i][fase[i]],c='g',s=200,marker='x',label='predication')
    plt.scatter(tru[i],seg[i][tru[i]],c='r',s=200,marker='+',label='label')
    plt.legend()
    plt.savefig('miss_extra.eps', dpi=600, format='eps')
    plt.show()

two=[]
three=[]
four=[]
for j in range(true_l_class_te.shape[0]):
    if len(set(true_l_class_te[j])) == 2:
        two.append(j)
    elif len(set(true_l_class_te[j])) == 3:
        three.append(j)
    elif len(set(true_l_class_te[j])) == 4:
        four.append(j)
model_name ='D:/python/multiple_label_new/model_t/class_res_' + str(148) + '_net.h5'
#model_name ='C:/Users/叶思超/Desktop/r峰结果/新建文件夹/1/class_res_' + str(129) + '_net.h5'
model = load_model(model_name)
pred_vt_r_t = model.predict(test_sig, verbose=1)
print(two)
print(three)
print(four)
plt.plot(test_sig[3352])
plt.axis('off')  #去掉坐标轴
plt.savefig('sig.eps', dpi=600, format='eps')
plt.show()

plt.plot(pred_vt_r_t[3352])
plt.axis('off')  #去掉坐标轴
plt.savefig('r_loc.eps', dpi=600, format='eps')
plt.show()
print(R_loc_te[3352])
print(true_l_R_te[3352])
print(true_l_class_te[3352])

rr=[]
for i in range(true_l_R_tr.shape[0]):
    for j in range(len(true_l_R_tr[0])-1):
        rr.append(true_l_R_tr[0][j+1]-true_l_R_tr[0][j])
rr_1=[]
for i in range(len(rr)):
    if rr[i]<87:
        rr_1.append(rr[i])
print(rr_1)
print(len(rr_1))'''
