from keras.models import load_model
from keras.utils import to_categorical
from utils import *
from focal_loss import *

classes=['R','FR']
true_l_R_tr = np.load('D:/python/multiple_label_new/npy_nds/true_label_R_train_seg.npy')
true_l_R_te = np.load('D:/python/multiple_label_new/npy_nds/true_label_R_test_seg.npy')
true_l_class_tr = np.load('D:/python/multiple_label_new/npy_nds/true_label_class_train_seg.npy')
true_l_class_te = np.load('D:/python/multiple_label_new/npy_nds/true_label_class_test_seg.npy')
train_sig = np.load('D:/python/multiple_label_new/npy_nds/beat_train_seg.npy')
test_sig = np.load('D:/python/multiple_label_new/npy_nds/beat_test_seg.npy')
label_R_tr = np.load('D:/python/multiple_label_new/npy_nds/label_R_peak_train_seg.npy')
label_R_te = np.load('D:/python/multiple_label_new/npy_nds/label_R_peak_test_seg.npy')
label_class_tr = np.load('D:/python/multiple_label_new/npy_nds/label_class_train_seg.npy')
label_class_te = np.load('D:/python/multiple_label_new/npy_nds/label_class_test_seg.npy')
global_class_tr = np.load('D:/python/multiple_label_new/npy_nds/global_class_train_seg.npy')
global_class_te = np.load('D:/python/multiple_label_new/npy_nds/global_class_test_seg.npy')
model_name ='D:/python/multiple_label_new/model_t/class_res_' + str(178) + '_net.h5'
#model_name ='C:/Users/叶思超/Desktop/r峰结果/新建文件夹/1/class_res_' + str(129) + '_net.h5'
model = load_model(model_name)
pred_vt_r = model.predict(train_sig, verbose=1)
pred_vt_r_t = model.predict(test_sig, verbose=1)
np.save('D:/python/multiple_label_new/pred/train_R_pre.npy',pred_vt_r)
np.save('D:/python/multiple_label_new/pred/test_R_pre.npy',pred_vt_r_t)
pred_vt_r=np.load('D:/python/multiple_label_new/pred/train_R_pre.npy')
pred_vt_r_t=np.load('D:/python/multiple_label_new/pred/test_R_pre.npy')


R_loc=R_precion(pred_vt_r.reshape(-1,1800), true_l_R_tr, classes,1800,'D:/python/multiple_label/pred/train_R_pre1.txt')
np.save('D:/python/multiple_label_new/feature_map/r_loc_tr.npy',R_loc)
R_loc=R_precion(pred_vt_r_t.reshape(-1,1800), true_l_R_te, classes,1800,'D:/python/multiple_label/pred/test_R_pre1.txt')
print(R_loc)
np.save('D:/python/multiple_label_new/feature_map/r_loc_te.npy',R_loc)
