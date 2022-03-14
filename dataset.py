import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import random
from scipy.ndimage.interpolation import rotate
import os
import scipy.io as sio
from datasetInfo import DatasetInfo

class HSIDataset:
    def __init__(self, data_name, pcaComponents=30, patchsz=5, is_pca=True, is_mirror=True, test_ratio=0.4):
        super().__init__()
        info = DatasetInfo.info[data_name]
        data = sio.loadmat(info['data_path'])[info['data_key']]
        label = sio.loadmat(info['label_path'])[info['label_key']]
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        if label.dtype != np.int32:
            label = label.astype(np.int32)
        self.data_name = data_name
        self.h, self.w, self.band = data.shape
        self.n_components = pcaComponents
        self.patchsz = patchsz
        assert self.patchsz != 0, 'patch size cant equal to 0!'
        self.dx = self.patchsz // 2
        self.test_ratio = test_ratio
        self.is_pca = is_pca
        self.is_mirror = is_mirror
        data = self.Normalize(data)
        if is_pca:
            data = self.applyPCA(data)
        data = self.padWithZeros(data)
        if is_mirror == True:
            data = self.addMirror(data)
        self.x_patch, self.y_patch = self.createPatches(data, label)
        x_train, x_test, y_train, y_test = self.trainTestSplit(self.x_patch, self.y_patch)
        x_train, y_train = self.augmentData(x_train, y_train)
        self.x_train, self.y_train, self.x_test, self.y_test = x_train, y_train, x_test, y_test
        self.save()

    def Normalize(self, x):
        new_x = x.reshape((-1, self.band))
        new_x -= np.min(new_x)
        new_x /= np.max(new_x)
        new_x = new_x.reshape((self.h, self.w, self.band))
        return new_x

    def applyPCA(self, x):
        new_x = np.reshape(x, (-1, self.band))
        pca = PCA(n_components=self.n_components, whiten=True)
        new_x = pca.fit_transform(new_x)
        new_x = np.reshape(new_x, (self.h, self.w, self.n_components))
        return new_x

    def padWithZeros(self, x):
        dx = self.dx
        new_x = np.zeros((self.h+2*dx, self.w+2*dx, x.shape[2]))
        new_x[dx:-dx, dx:-dx] = x
        return new_x

    def addMirror(self, x):
        dx = self.dx
        assert x.shape == (self.h+2*dx, self.w+2*dx, x.shape[2])
        for i in range(dx):
            x[i, :, :] = x[2*dx-i, :, :]
            x[:, i, :] = x[:, 2*dx-i, :]
            x[-i-1, :, :] = x[-(2*dx-i)-1, :, :]
            x[:, -i-1, :] = x[:, -(2*dx-i)-1, :]
        return x

    def createPatches(self, data, label):
        nonzero = np.nonzero(label)
        sample_ind = list(zip(*nonzero))
        num_sample = len(sample_ind)
        patched_data = np.zeros((num_sample, self.patchsz, self.patchsz, data.shape[2]))
        patched_label = np.zeros(num_sample)
        for i, (x, y) in enumerate(sample_ind):
            patched_data[i] = data[x:x+2*self.dx+1, y:y+2*self.dx+1]
            patched_label[i] = label[x, y] - 1
        return patched_data, patched_label

    def trainTestSplit(self, x_patch, y_patch):
        x_train, x_test, y_train, y_test = train_test_split(x_patch, y_patch,
                                                            test_size=self.test_ratio,
                                                            random_state=123, stratify=y_patch)
        return x_train, x_test, y_train, y_test

    def augmentData(self, x_train, y_train):
        temp = np.copy(x_train)
        y_temp = np.copy(y_train)
        for i in range(temp.shape[0]):
            rand_ind = random.randint(0, 4)
            if rand_ind == 0:
                temp[i] = np.fliplr(temp[i])
            elif rand_ind == 1:
                temp[i] = np.flipud(temp[i])
            elif rand_ind == 2:
                temp[i] = rotate(temp[i], angle=90, reshape=False, prefilter=False)
            elif rand_ind == 3:
                temp[i] = rotate(temp[i], angle=180, reshape=False, prefilter=False)
            else:
                temp[i] = rotate(temp[i], angle=270, reshape=False, prefilter=False)
        x_train = np.concatenate((x_train, temp))
        y_train = np.concatenate((y_train, y_temp))
        return x_train, y_train

    def save(self):
        if self.is_pca:
            path = './predata/{}+npca{}+patchsz{}+testsize{}'.format(self.data_name, self.n_components,
                                                                     self.patchsz, self.test_ratio)
        else:
            path = './predata/{}+npca{}+testsize{}'.format(self.data_name, self.patchsz, self.test_ratio)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, 'x_train.npy'), 'bw') as f:
            np.save(f, self.x_train)
        with open(os.path.join(path, 'y_train.npy'), 'bw') as f:
            np.save(f, self.y_train)
        with open(os.path.join(path, 'x_test.npy'), 'bw') as f:
            np.save(f, self.x_test)
        with open(os.path.join(path, 'y_test.npy'), 'bw') as f:
            np.save(f, self.y_test)
        with open(os.path.join(path, 'x_all.npy'), 'bw') as f:
            np.save(f, self.x_patch)
        with open(os.path.join(path, 'y_all.npy'), 'bw') as f:
            np.save(f, self.y_patch)
        print('HSI data preprocessing finished!')




























