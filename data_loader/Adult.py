import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import data_loader.data_utils as utils
import os


class AdultDataset(Dataset):
    def __init__(self, path, mode, quick_load, bias_name = None, training_number = 40000):
        self.bias_name = bias_name
        self.features, self.labels_onehotv, self.bias_onehotv = self.load_data(path, quick_load)
        self.labels = self.onehotvector_to_label(self.labels_onehotv)
        self.bias = self.bias_onehotv
        self.bias_name = bias_name
        if mode == 'train':
            self.features = self.features[:training_number]
            self.labels = self.labels[:training_number]
            self.bias = self.bias[:training_number]
        elif mode == 'validation':
            self.features = self.features[training_number:]
            self.labels = self.labels[training_number:]
            self.bias = self.bias[training_number:]
        elif mode == 'all':
            pass


    def __getitem__(self, i):
        feature = self.features[i]
        label = self.labels[i]
        label = np.array(label)
        bias = self.bias[i]
        return feature, label, bias

    def __len__(self):
        data_len = self.features.shape[0]
        return data_len

    def load_data(self, path ,quick_load):
        if quick_load:
            data = utils.quick_load(path)
        else:
            data = utils.data_processing(path)
        data = self.data_preproccess(data)
        features = data[:,:data.shape[1]-2]
        labels = data[:,data.shape[1]-2:]
        bias = utils.get_bias(data, self.bias_name)
        return features, labels, bias

    def data_preproccess(self, data):
        '''
        :param data:
        :return:
        '''
        data = torch.tensor(data,dtype=torch.float)
        return data

    def onehotvector_to_label(self, onehotvector):
        labels = []
        for v in onehotvector:
            for i in range(v.shape[0]):
                if v[i] != 0:
                    label = i
                    labels.append(label)
                    break
        labels = torch.tensor(labels, dtype=torch.long)
        return labels


if __name__ == '__main__':
    load_path = "./raw_data/adult.csv"
    quick_load_path = "./raw_data/newData.csv"
    dtrain = torch.utils.data.DataLoader(
        AdultDataset(
            path = quick_load_path,
            mode='train',
            quick_load = True,
            bias_name = 'relationship'
        ),
        batch_size=50,
        shuffle=False
    )

    x, y, bias= next(iter(dtrain))
    print(x.shape, x.min(), x.max())
    print(y.shape, y.min(), y.max())

    print('size:', len(dtrain))