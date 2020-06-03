import warnings
from utils import *
from fin_net import *
from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback,ReduceLROnPlateau
from sklearn.metrics import roc_curve, auc
from keras.utils import to_categorical
from keras.models import load_model
from scipy import interp
warnings.filterwarnings("ignore")
# class_num=[43252,797,3179,359]
class_weights = {}
sig_class = ['N', 'S', 'V', 'F']

feature_sum_tr=np.load('D:/python/multiple_label_new/fin_feature/fin_feature_3_tr.npy')
#feature_sum_tr=np.expand_dims(feature_sum_tr, axis=3)
label_sum_tr=np.load('D:/python/multiple_label_new/fin_feature/fin_label_tr.npy')
label_sum_tr=to_categorical(label_sum_tr, num_classes=len(sig_class))
feature_sum_te=np.load('D:/python/multiple_label_new/fin_feature/fin_feature_3_te.npy')
#feature_sum_te=np.expand_dims(feature_sum_te, axis=3)
label_sum_te=np.load('D:/python/multiple_label_new/fin_feature/fin_label_te.npy')
label_sum_te=to_categorical(label_sum_te, num_classes=len(sig_class))
'''model_name = 'C:/Users/叶思超/Desktop/r峰结果/新建文件夹/1/3/class_res1_' + str(97) + '_net.h5'
#model_name = 'D:/python/multiple_label_new/fin_model/class_res1_' + str(51) + '_net.h5'
model_name = 'D:/python/multiple_label_new/fin_model/my_model_1.hdf5'
model = load_model(model_name)
pred_vt = model.predict(feature_sum_tr, batch_size=128, verbose=0)
np.save('pred_vr.npy',pred_vt)
print(pred_vt)
roc_probs = np.ndarray.sum(pred_vt, axis=1)
print(roc_probs)
print(roc_probs.shape)
pred_v = np.argmax(pred_vt, axis=1)
true_v = np.argmax(label_sum_tr, axis=1)

plot_confusion_matrix(true_v, pred_v, np.array(sig_class))
print_results(true_v, pred_v, sig_class, 'sen_p+_1.txt')
plt.show()

pred_vt = model.predict(feature_sum_te, batch_size=128, verbose=0)'''
#pred_vt =np.load('C:/Users/叶思超/Desktop/r峰结果/新建文件夹/1/6/pred_vt.npy')
pred_vt=np.load('pred_vt.npy')
print(pred_vt)
roc_probs = np.ndarray.sum(pred_vt, axis=1)
print(roc_probs)
print(roc_probs.shape)
pred_v = np.argmax(pred_vt, axis=1)
true_v = label_sum_te
print(true_v.shape,pred_vt.shape)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(4):
    fpr[i], tpr[i], _ = roc_curve(true_v[:, i], pred_vt[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(true_v.ravel(), pred_vt.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(4)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(4):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
# Finally average it and compute AUC
mean_tpr /= 4
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

lw=2
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.4f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = ['aqua', 'darkorange', 'cornflowerblue','g']
print(roc_auc[3])
for i in range(4):
    print(i)
    plt.plot(fpr[i], tpr[i], color=colors[i], lw=lw,
             label='ROC curve of '+sig_class[i]+' (area = {0:0.4f})'.format(roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.savefig('Roc.eps', dpi=600, format='eps')
plt.show()
