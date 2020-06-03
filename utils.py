from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
import numpy as np
import pywt
import math
cpu_threads = 7
def sig_wt_filt(sig):
    """
    对信号进行小波变换滤波
    :param sig: 输入信号，1-d array
    :return: 小波滤波后的信号，1-d array

    """
    sig = sig.reshape(sig.shape[0])
    coeffs = pywt.wavedec(sig, 'db8', level=9)
    coeffs[-1] = np.zeros(len(coeffs[-1]))
    coeffs[-2] = np.zeros(len(coeffs[-2]))
    coeffs[0] = np.zeros(len(coeffs[0]))
    sig_filt = pywt.waverec(coeffs, 'db6')
    return sig_filt

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    绘制混淆矩阵图，来源：
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix'#, without normalization

    cm = confusion_matrix(y_true, y_pred)
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def print_results(y_true, y_pred, target_names,filename):
    """
    打印相关结果
    :param y_true: 期望输出，1-d array
    :param y_pred: 实际输出，1-d array
    :param target_names: 各类别名称
    :return: 打印结果
    """
    overall_accuracy = accuracy_score(y_true, y_pred)
    print('\n----- overall_accuracy: {0:f} -----'.format(overall_accuracy))
    cm = confusion_matrix(y_true, y_pred)
    f = open(filename, "w")
    f.write("Confusion Matrix:" + "\n\n")
    f.write(str(cm)+ "\n\n")
    f.write('----- overall_accuracy: {0:f} -----\n'.format(overall_accuracy))
    for i in range(len(target_names)):
        print(target_names[i] + ':')
        Se = cm[i][i]/np.sum(cm[i])
        Pp = cm[i][i]/np.sum(cm[:, i])
        print('  Se = ' + str(Se))
        print('  P+ = ' + str(Pp))
        if i==0:
            se_mean=Se
            Pp_mean= Pp
        else:
            se_mean = Se+se_mean
            Pp_mean = Pp+Pp_mean
        f.write(target_names[i] + ':'+ "\n")
        f.write('  Se = ' + str(Se) + "\n")
        f.write('  P+ = ' + str(Pp) + "\n")
    print('  Se_mean = ' + str(se_mean/4))
    print('  P+ = ' + str(Pp_mean/4))
    print('  F1 = ' + str(2 * (Pp_mean * se_mean/4) / (Pp_mean + se_mean)))
    print('--------------------------------------')
    f.close()

def R_precion(pred_vt_r,true_l_R_te,classes,seg_lenth,filename):
    TP=0
    FP=0
    FN=0
    TN=0
    r_loction=[]
    R_loc=[]
    for i in range(pred_vt_r.shape[0]):
        mid_l_class=[]
        j=0
        while j < pred_vt_r.shape[1]:
            if pred_vt_r[i][j]>0.2 :
                if j+1<pred_vt_r.shape[1]:
                    if pred_vt_r[i][j]>np.max(pred_vt_r[i][j+1:j+87]):
                        mid_l_class.append(j)
                    else:
                        mid_l_class.append(np.argmax(pred_vt_r[i][j+1:j+87])+j+1)
                j=j+87
            else:
                j=j+1
        FP=FP+len(mid_l_class)
        FN=FN+len(true_l_R_te[i])
        r_loction=mid_l_class[:]
        R_loc.append(r_loction)
        for m in range(len(true_l_R_te[i])):
            for s in range(len(mid_l_class)):
                if math.fabs(mid_l_class[s]-true_l_R_te[i][m])<0.15*360:
                    TN=TN+1
                    del mid_l_class[s]
                    break
    TP=pred_vt_r.shape[0]*seg_lenth-FN
    FN=FN-TN
    FP=FP-TN
    TP=TP-FP
    cm=np.zeros((2,2))
    cm[0][0]=TN#写反了，算了，懒得改  TP   FN
    cm[0][1]=FN#                     FP   TN
    cm[1][1]=TP
    cm[1][0]=FP
    cm= np.asarray(cm,dtype=int)
    title = 'Confusion matrix'
    fig, ax = plt.subplots()
    cmap=plt.cm.Blues
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    #plt.savefig(plt_name+'.eps', dpi=600, format='eps')
    plt.show()

    print(cm)
    f = open(filename, "w")
    f.write("Confusion Matrix:" + "\n\n")
    f.write(str(cm)+ "\n\n")
    overall_accuracy=(TN)/(TN+FP+FN)
    f.write('----- overall_accuracy: {0:f} -----\n'.format(overall_accuracy))
    print('----- overall_accuracy: {0:f} -----\n'.format(overall_accuracy))
    for i in range(len(classes)):
        print(classes[i] + ':')
        Se = cm[i][i]/np.sum(cm[i])
        Pp = cm[i][i]/np.sum(cm[:, i])
        print('  Se = ' + str(Se))
        print('  P+ = ' + str(Pp))
        if i==0:
            se_mean=Se
            Pp_mean= Pp
        else:
            se_mean = Se+se_mean
            Pp_mean = Pp+Pp_mean
        f.write(classes[i] + ':'+ "\n")
        f.write('  Se = ' + str(Se) + "\n")
        f.write('  P+ = ' + str(Pp) + "\n")
    print('  Se_mean = ' + str(se_mean/2))
    print('  P+ = ' + str(Pp_mean/2))
    print('--------------------------------------')
    f.close()
    return R_loc
def Proofreading(pred_fin,true_l_class,true_l_R,R_loc,classes,filename):
    cm = np.zeros((5,5))
    for j in range(true_l_class.shape[0]):
        class_lab=true_l_class[j][:].tolist()
        r_lab=true_l_R[j][:].tolist()
        class_pre=pred_fin[j][:]
        r_loc=R_loc[j][:]
        num=0
        #print(class_lab,class_pre,r_loc,r_lab)
        for s in range(len(true_l_R[j])):
            for i in range(len(r_loc)):
                if math.fabs(r_loc[i]-true_l_R[j][s])<0.15*360:
                    cm[true_l_class[j][s]][pred_fin[j][i]]=cm[true_l_class[j][s]][pred_fin[j][i]]+1
                    class_lab[s]=-1
                    del r_loc[i],class_pre[i]
                    break
        for m in range(len(class_lab)):
            if class_lab[m]==-1:
                num=num+1
        for m in range(num):
            class_lab.remove(-1)
        for m in range(len(class_lab)):
            cm[4][class_lab[m]] = cm[4][class_lab[m]] + 1
        for m in range(len(class_pre)):
            cm[class_pre[m]][4] = cm[class_pre[m]][4] + 1
    cm = cm.astype(np.int)
    print(cm)
    title = 'Confusion matrix'
    fig, ax = plt.subplots()
    cmap = plt.cm.Blues
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    x_classes=['N','S','V','F','extra']
    y_classes = ['N', 'S', 'V', 'F', 'miss']
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=x_classes, yticklabels=y_classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
    cm = cm[0:4,0:4]
    overall_accuracy =(cm[0][0]+cm[1][1]+cm[2][2]+cm[3][3])/np.sum(cm)
    print('\n----- overall_accuracy: {0:f} -----'.format(overall_accuracy))
    f = open(filename, "w")
    f.write("Confusion Matrix:" + "\n\n")
    f.write(str(cm)+ "\n\n")
    f.write('----- overall_accuracy: {0:f} -----\n'.format(overall_accuracy))
    for i in range(len(classes)):
        print(classes[i] + ':')
        Se = cm[i][i]/np.sum(cm[i])
        Pp = cm[i][i]/np.sum(cm[:, i])
        print('  Se = ' + str(Se))
        print('  P+ = ' + str(Pp))
        if i==0:
            se_mean=Se
            Pp_mean= Pp
        else:
            se_mean = Se+se_mean
            Pp_mean = Pp+Pp_mean
        f.write(classes[i] + ':'+ "\n")
        f.write('  Se = ' + str(Se) + "\n")
        f.write('  P+ = ' + str(Pp) + "\n")
    print('  Se_mean = ' + str(se_mean/4))
    print('  P+ = ' + str(Pp_mean/4))
    print('--------------------------------------')
    f.close()

'''a=99.5557
b=99.6181
c=99.5970
print((a+b+c)/3)'''
