import os.path
import numpy as np
from tensorflow.keras import backend as BK
BK.set_image_data_format('channels_last')
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.utils import to_categorical
from model.module import CNN
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)

def load_data(path):
    x_train = np.load(os.path.join(path, 'x_train.npy'))
    y_train = np.load(os.path.join(path, 'y_train.npy'))
    x_test = np.load(os.path.join(path, 'x_test.npy'))
    y_test = np.load(os.path.join(path, 'y_test.npy'))
    return x_train, y_train, x_test, y_test

def train(predata_name):
    data_path = os.path.join('./predata', predata_name)
    x_train, y_train, x_test, y_test = load_data(data_path)
    x_train, x_test = x_train.astype(np.float32), x_test.astype(np.float32)
    y_train, y_test = y_train.astype(np.int32), y_test.astype(np.int32)
    num_class = np.max(y_train) + 1
    input_shape = x_train[0].shape
    in_channel = x_train[0].shape[2]

    model = CNN(input_shape, in_channel, num_class)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=25,
                                  min_lr=1e-6, verbose=1)
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss=SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['sparse_categorical_accuracy'])

    model_path = './model/{}/checkpoint/CNN.ckpt'.format(predata_name)
    if os.path.exists(model_path + '.index'):
        print('-----------------loading model---------------------')
        model.load_weights(model_path)
    checkpointer = ModelCheckpoint(filepath=model_path, verbose=1, save_best_only=True)

    history = model.fit(x_train, y_train,
                        batch_size=32,
                        epochs=10,
                        verbose=1,
                        validation_data=(x_test, y_test),
                        callbacks=[reduce_lr, checkpointer],
                        shuffle=True)
    model.summary()
    file = open('./model/{}/params.txt'.format(predata_name), 'w')
    for v in model.trainable_variables:
        file.write(str(v.name) + '\n')
        file.write(str(v.shape) + '\n')
        file.write(str(v.numpy()) + '\n')
    file.close()
    # TODO:solve the issue of plot model
    acc = history.history['sparse_categorical_accuracy']
    val_acc = history.history['val_sparse_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    train('salinas+npca30+patchsz5+testsize0.4')