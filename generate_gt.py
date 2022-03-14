import spectral
import matplotlib.pyplot as plt
from scipy.io import loadmat
from datasetInfo import DatasetInfo
import numpy as np

def generate_gt_map(data_name):
    save_path = './gt/{}_gt.jpg'.format(data_name)
    info = DatasetInfo.info[data_name]
    data = loadmat(info['data_path'])[info['data_key']]
    data = data.astype(np.float32)
    label = loadmat(info['label_path'])[info['label_key']]
    label = label.astype(np.int32)
    spectral.imshow(classes=label, figsize=(10, 10))
    plt.savefig(save_path)

if __name__ == '__main__':
    generate_gt_map('salinas')