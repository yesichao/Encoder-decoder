import warnings
from utils import *
from fin_net import *
from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback,ReduceLROnPlateau
from keras.utils import to_categorical
warnings.filterwarnings("ignore")
# class_num=[43252,797,3179,359]
class_weights = {}
sig_class = ['N', 'S', 'V', 'F']
class Metrics(Callback):
    def on_epoch_end(self, epoch, logs={}):
        pred_vt = model.predict(feature_sum_te, batch_size=128, verbose=0)
        print(pred_vt)
        roc_probs = np.ndarray.sum(pred_vt, axis=1)
        print(roc_probs)
        print(roc_probs.shape)
        np.save('D:/python/multiple_label_new/' + 'pred' + '_' + 'test' + '.npy', pred_vt)
        pred_v = np.argmax(pred_vt, axis=1)
        true_v = np.argmax(label_sum_te, axis=1)

        plot_confusion_matrix(true_v, pred_v, np.array(sig_class))
        print_results(true_v, pred_v, sig_class,'sen_p+_1.txt')
        plt.show()
        model.save('D:/python/multiple_label_new/fin_model/class_res1_' + str(epoch) + '_net.h5')
        return
metrics = Metrics()
map_num=3
feature_sum_tr=np.load('D:/python/multiple_label_new/fin_feature/fin_feature_'+str(map_num)+'_tr.npy')
#feature_sum_tr=np.expand_dims(feature_sum_tr, axis=3)
label_sum_tr=np.load('D:/python/multiple_label_new/fin_feature/fin_label_tr.npy')
'''Indices = np.arange(feature_sum_tr.shape[0])  # 随机打乱索引
np.random.shuffle(Indices)
feature_sum_tr=feature_sum_tr[Indices]
label_sum_tr=label_sum_tr[Indices]
for c in range(4):
    class_weights.update({c: label_sum_tr.shape[0]/float(np.count_nonzero(label_sum_tr== c))})'''
label_sum_tr=to_categorical(label_sum_tr, num_classes=len(sig_class))
feature_sum_te=np.load('D:/python/multiple_label_new/fin_feature/fin_feature_'+str(map_num)+'_te.npy')
#feature_sum_te=np.expand_dims(feature_sum_te, axis=3)
label_sum_te=np.load('D:/python/multiple_label_new/fin_feature/fin_label_te.npy')
label_sum_te=to_categorical(label_sum_te, num_classes=len(sig_class))

model = Net(map_num)
MODEL_PATH = 'D:/python/multiple_label_new/fin_model/'
model_name = 'my_model_' + str(1) + '.hdf5'

checkpoint = ModelCheckpoint(filepath=MODEL_PATH + model_name,
                             monitor='loss', mode='min',
                             save_best_only='True')
callback_lists = [checkpoint,metrics]
model.fit(x=feature_sum_tr, y=label_sum_tr, batch_size=128, epochs=100, class_weight=class_weights,verbose=1,
          validation_data=(feature_sum_te, label_sum_te), callbacks=callback_lists )