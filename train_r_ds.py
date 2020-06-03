from dense_net import *
import warnings
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam
from focal_loss import *
from utils import *
from keras.callbacks import Callback
from keras.utils import to_categorical
#from r_peak_net import *

warnings.filterwarnings("ignore")
# class_num=[43252,797,3179,359]
class_num = [1, 1, 1, 1]
sig_class = ['N', 'S', 'V', 'F']
classes = ['R', 'FR']

class Metrics(Callback):
    def on_epoch_end(self, epoch, logs={}):
        pred_vt_r = model.predict(train_sig, verbose=0)
        print('calculation train ...')
        pred_fin = []
        R_loc = R_precion(pred_vt_r.reshape(-1, 1800), true_l_R_tr, classes,1800,
                          'D:/python/multiple_label_new/result/train_'+str(epoch)+'.txt')

        pred_vt_r = model.predict(test_sig, verbose=0)
        print('calculation...')
        pred_fin = []
        R_loc = R_precion(pred_vt_r.reshape(-1, 1800), true_l_R_te, classes,1800,
                          'D:/python/multiple_label_new/result/test_'+str(epoch)+'.txt')
        model.save('D:/python/multiple_label_new/model_t/class_res_' + str(epoch) + '_net.h5')
        return


metrics = Metrics()


def smooth_L1_loss(y_true, y_pred):
    THRESHOLD = K.variable(1.0)
    mae = K.abs(y_true - y_pred)
    flag = K.greater(mae, THRESHOLD)
    loss = K.mean(K.switch(flag, (mae - 0.5), K.pow(mae, 2)), axis=-1)
    return loss


def lr_schedule(epoch):
    # 训练网络时学习率衰减方案
    lr = 0.001
    if epoch >= 20:
        lr = 0.0001
    elif epoch >= 40:
        lr = 0.00001
    elif epoch >= 70:
        lr = 0.000001
    print('Learning rate: ', lr)
    return lr


true_l_R_tr = np.load('D:/python/multiple_label_new/npy/true_label_R_train_seg.npy')
true_l_R_te = np.load('D:/python/multiple_label_new/npy/true_label_R_test_seg.npy')
true_l_class_tr = np.load('D:/python/multiple_label_new/npy/true_label_class_train_seg.npy')
true_l_class_te = np.load('D:/python/multiple_label_new/npy/true_label_class_test_seg.npy')
train_sig = np.load('D:/python/multiple_label_new/npy/beat_train_seg.npy')
test_sig = np.load('D:/python/multiple_label_new/npy/beat_test_seg.npy')
label_R_tr = np.load('D:/python/multiple_label_new/npy/label_R_peak_train_seg.npy')
label_R_te = np.load('D:/python/multiple_label_new/npy/label_R_peak_test_seg.npy')
label_class_tr = np.load('D:/python/multiple_label_new/npy/label_class_train_seg.npy')
label_class_te = np.load('D:/python/multiple_label_new/npy/label_class_test_seg.npy')
global_class_tr = np.load('D:/python/multiple_label_new/npy/global_class_train_seg.npy')
global_class_te = np.load('D:/python/multiple_label_new/npy/global_class_test_seg.npy')
label_R_tr = np.expand_dims(label_R_tr, axis=2)
label_R_te = np.expand_dims(label_R_te, axis=2)
Indices = np.arange(train_sig.shape[0])  # 随机打乱索引
np.random.shuffle(Indices)
train_sig = train_sig[Indices]
label_R_tr = label_R_tr[Indices]
label_class_tr = label_class_tr[Indices]
global_class_tr=global_class_tr[Indices]
true_l_R_tr=true_l_R_tr[Indices]
true_l_class_tr=true_l_class_tr[Indices]

print(train_sig.shape, test_sig.shape,
      label_R_tr.shape, label_R_te.shape,
      label_class_tr.shape, label_class_te.shape, global_class_tr.shape, global_class_te.shape)
model = Dense_Net()
#model = Net()
MODEL_PATH = 'D:/python/multiple_label_new/model_t/'
model_name = 'my_model_' + str(1) + '.hdf5'
optimizer = Adam(lr_schedule(0))
model.compile(optimizer=optimizer,
              loss={'r_output': 'binary_crossentropy'
                    })

checkpoint = ModelCheckpoint(filepath=MODEL_PATH + model_name,
                             monitor='loss', mode='min',
                             save_best_only='True')
model.fit(x=train_sig, y={'r_output': label_R_tr}, batch_size=32, epochs=100, verbose=1,
          validation_data=(test_sig, {'r_output': label_R_te}), callbacks=[checkpoint, metrics])