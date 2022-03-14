import os.path
from model.module import CNN
import scipy.io as sio
from datasetInfo import DatasetInfo
import numpy as np
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import itertools
from sklearn.decomposition import PCA
import spectral


def load_model(predata_name):
    path = os.path.join('./predata', predata_name)
    x_test = np.load(os.path.join(path, 'x_test.npy'))
    y_test = np.load(os.path.join(path, 'y_test.npy'))
    input_shape = x_test[0].shape
    in_channel = x_test[0].shape[2]
    num_class = np.max(y_test) + 1
    model = CNN(input_shape=input_shape, in_channel=in_channel, num_class=num_class)
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss=SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['sparse_categorical_accuracy'])
    model_path = './model/{}/checkpoint/CNN.ckpt'.format(predata_name)
    if os.path.exists(model_path + '.index'):
        print('-----------------loading model---------------------')
        model.load_weights(model_path)
    return model

def report(model, x_test, y_test, target_names):
    pred = model.predict(x_test)
    pred = np.argmax(pred, axis=1)
    class_acc = classification_report(y_test, pred, target_names=target_names)
    confusion_mat = confusion_matrix(y_test, pred)
    score = model.evaluate(x_test, y_test, batch_size=32)
    test_loss = score[0]
    test_acc = score[1] * 100
    return class_acc, confusion_mat, test_loss, test_acc

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.get_cmap("Blues")):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    Normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    if normalize:
        cm = Normalized
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(Normalized, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.4f' if normalize else 'd'
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        thresh = cm[i].max() / 2.
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def test(data_name, predata_name, model):
    info = DatasetInfo.info[data_name]
    target_names = info['target_names']
    path = os.path.join('./predata', predata_name)
    x_test = np.load(os.path.join(path, 'x_test.npy')).astype(np.float32)
    y_test = np.load(os.path.join(path, 'y_test.npy')).astype(np.int32)
    class_acc, cm, test_loss, test_acc = report(model, x_test, y_test, target_names)
    class_acc = str(class_acc)
    cm_str = str(cm)
    print('Test loss:{}'.format(test_loss))
    print('Test acc:{}%'.format(test_acc))
    print('Classification result:')
    print(class_acc)
    print('Confusion matrix:')
    print(cm_str)

    report_save_path = os.path.join('./result', predata_name)
    if not os.path.exists(report_save_path):
        os.makedirs(report_save_path)
    file_name = os.path.join(report_save_path, 'report.txt')
    with open(file_name, 'w') as f:
        f.write('Test loss:{}'.format(test_loss))
        f.write('\n')
        f.write('Test acc:{}%'.format(test_acc))
        f.write('\n')
        f.write('\n')
        f.write('Classification result:\n')
        f.write('{}'.format(class_acc))
        f.write('\n')
        f.write('Confusion matrix:\n')
        f.write('{}'.format(cm_str))
        f.write('\n')
    print('-------------successfully create report.txt!-------------------')

    plt.figure(figsize=(15, 15))
    plot_confusion_matrix(cm, classes=target_names, normalize=False,
                          title='Confusion matrix, without normalization')
    plt.savefig(os.path.join(report_save_path, 'confusion_mat_without_norm.png'))
    print('------------succesfully generate confusion matrix pic!-----------')



#TODO:加一个将模型存放在不同文件夹的功能
def generate_predmap(data_name, predata_name, model, pcaComponents=30, patchsz=5, is_mirror=True, is_pca=True):
    info = DatasetInfo.info[data_name]
    data = sio.loadmat(info['data_path'])[info['data_key']]
    data = data.astype(np.float32)
    label = sio.loadmat(info['label_path'])[info['label_key']]
    label = label.astype(np.int32)
    x_all = np.load('./predata/{}/x_all.npy'.format(predata_name))
    y_all = np.load('./predata/{}/y_all.npy'.format(predata_name))
    nonzero = np.nonzero(label)
    sample_ind = list(zip(*nonzero))
    num_sample = len(sample_ind)
    pred_map = np.zeros_like(label)
    pred = model.predict(x_all, batch_size=32)
    pred = np.argmax(pred, axis=1)
    for i, (x, y) in enumerate(sample_ind):
        pred_map[x, y] = pred[i] + 1
    pred_pic_savepath = './result/{}'.format(predata_name)
    if not os.path.exists(pred_pic_savepath):
        os.makedirs(pred_pic_savepath)
    predict_image = spectral.imshow(classes=pred_map.astype(int), figsize=(10, 10))
    plt.savefig(os.path.join(pred_pic_savepath, 'pred_map.jpg'))
    print('------------------sucessfully saved pred map!----------------------')

if __name__ == '__main__':
    model = load_model('salinas+npca30+patchsz5+testsize0.4')
    test('salinas', 'salinas+npca30+patchsz5+testsize0.4', model)
    generate_predmap('salinas', 'salinas+npca30+patchsz5+testsize0.4', model)