from mask_cnn import *
from utils import *
sig_class = ['N', 'S', 'V', 'F']
classes = ['R', 'FR']
true_l_R_tr = np.load('D:/python/multiple_label_new/npy_nds/1/true_label_R_train_seg.npy')
true_l_R_te = np.load('D:/python/multiple_label_new/npy_nds/1/true_label_R_test_seg.npy')
true_l_class_tr = np.load('D:/python/multiple_label_new/npy_nds/1/true_label_class_train_seg.npy')
true_l_class_te = np.load('D:/python/multiple_label_new/npy_nds/1/true_label_class_test_seg.npy')
train_sig = np.load('D:/python/multiple_label_new/npy_nds/1/beat_train_seg.npy')
test_sig = np.load('D:/python/multiple_label_new/npy_nds/1/beat_test_seg.npy')
label_R_tr = np.load('D:/python/multiple_label_new/npy_nds/1/label_R_peak_train_seg.npy')
label_R_te = np.load('D:/python/multiple_label_new/npy_nds/1/label_R_peak_test_seg.npy')
label_class_tr = np.load('D:/python/multiple_label_new/npy_nds/1/label_class_train_seg.npy')
label_class_te = np.load('D:/python/multiple_label_new/npy_nds/1/label_class_test_seg.npy')
global_class_tr = np.load('D:/python/multiple_label_new/npy_nds/1/global_class_train_seg.npy')
global_class_te = np.load('D:/python/multiple_label_new/npy_nds/1/global_class_test_seg.npy')
label_R_tr = np.expand_dims(label_R_tr, axis=2)
label_R_te = np.expand_dims(label_R_te, axis=2)

print(train_sig.shape, test_sig.shape,
      label_R_tr.shape, label_R_te.shape,
      label_class_tr.shape, label_class_te.shape, global_class_tr.shape, global_class_te.shape)
model = Net()
feature_map_1,feature_map_2,feature_map_3,feature_map_4,feature_map_5,feature_map_6,feature_map_7,r_output = model.predict(train_sig, verbose=0)
R_loc_tr = R_precion(r_output.reshape(-1, 1800), true_l_R_tr, classes, 1800,
                  'D:/python/multiple_label_new/result/train.txt')
np.save('D:/python/multiple_label_new/feature_map/feature_map_1_tr.npy',feature_map_1)
np.save('D:/python/multiple_label_new/feature_map/feature_map_2_tr.npy',feature_map_2)
np.save('D:/python/multiple_label_new/feature_map/feature_map_3_tr.npy',feature_map_3)
np.save('D:/python/multiple_label_new/feature_map/feature_map_4_tr.npy',feature_map_4)
np.save('D:/python/multiple_label_new/feature_map/feature_map_5_tr.npy',feature_map_5)
np.save('D:/python/multiple_label_new/feature_map/feature_map_6_tr.npy',feature_map_6)
np.save('D:/python/multiple_label_new/feature_map/feature_map_7_tr.npy',feature_map_7)
np.save('D:/python/multiple_label_new/feature_map/r_loc_tr.npy',R_loc_tr)
'''feature_map_1,feature_map_2,feature_map_3,feature_map_4,feature_map_5,feature_map_6,feature_map_7,r_output = model.predict(test_sig, verbose=0)
R_loc_te = R_precion(r_output.reshape(-1, 1800), true_l_R_te, classes, 1800,
                  'D:/python/multiple_label_new/result/train.txt')
np.save('D:/python/multiple_label_new/feature_map/feature_map_1_te.npy',feature_map_1)
np.save('D:/python/multiple_label_new/feature_map/feature_map_2_te.npy',feature_map_2)
np.save('D:/python/multiple_label_new/feature_map/feature_map_3_te.npy',feature_map_3)
np.save('D:/python/multiple_label_new/feature_map/feature_map_4_te.npy',feature_map_4)
np.save('D:/python/multiple_label_new/feature_map/feature_map_5_te.npy',feature_map_5)
np.save('D:/python/multiple_label_new/feature_map/feature_map_6_te.npy',feature_map_6)
np.save('D:/python/multiple_label_new/feature_map/feature_map_7_te.npy',feature_map_7)
np.save('D:/python/multiple_label_new/feature_map/r_loc_te.npy',R_loc_te)'''