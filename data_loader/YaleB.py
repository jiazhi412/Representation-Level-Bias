import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from PIL import Image
from sklearn import preprocessing


def encodeColumn(oldCol, encoder):
    newCol = []
    for c in oldCol:
        c_array = np.array(c).reshape(-1, 1)
        newCol.append(encoder.transform(c_array).toarray())
    return np.array(newCol)


def bias_encoding(bias):
    enc = preprocessing.OneHotEncoder()
    enc.fit(bias.reshape((-1, 1)))
    bias = encodeColumn(bias, enc)
    bias = bias.reshape((-1, 5))
    return bias


class YaleBDataset(Dataset):
    def __init__(self, path, mode, training_number = 40000):
        self.images, self.labels, self.bias = self.load_data(path)
        self.to_tensor()
        training_index, testing_index = self.dataset_split()
        self.bias = bias_encoding(self.bias)
        if mode == 'train':
            self.images, self.labels, self.bias = self.images[training_index], self.labels[training_index], self.bias[training_index]
        elif mode == 'test':
            self.images, self.labels, self.bias = self.images[testing_index], self.labels[testing_index], self.bias[testing_index]
        elif mode == 'all':
            pass

    def __getitem__(self, i):
        image = self.images[i]
        label = self.labels[i]
        bias = self.bias[i]
        return image, label, bias

    def __len__(self):
        data_len = self.images.shape[0]
        return data_len

    def load_data(self, root, reduce=4):
        """
        Load ORL (or Extended YaleB) dataset to numpy array.

        Args:
            root: path to dataset.
            reduce: scale factor for zooming out images.

        """
        images, labels, bias = [], [], []

        for i, person in enumerate(sorted(os.listdir(root))):

            if not os.path.isdir(os.path.join(root, person)):
                continue

            for fname in os.listdir(os.path.join(root, person)):

                # Remove background images in Extended YaleB dataset.
                if fname.endswith('Ambient.pgm'):
                    continue

                if not fname.endswith('.pgm'):
                    continue

                def extract_lighting(fname):
                    lighting_text = fname[12:16]
                    lighting_value = int(lighting_text)

                    lighting_res_label = 0
                    # front -> 1 ; upper_right -> 2 ; upper_left -> 3 ; lower_right -> 4 ; lower_left -> 5
                    if lighting_value >= -30 and lighting_value <= 30:
                        lighting_res_label = 1
                    elif lighting_value >= -90 and lighting_value < -30:
                        lighting_res_label = 2
                    elif lighting_value < -90:
                        lighting_res_label = 4
                    elif lighting_value > 30 and lighting_value <= 90:
                        lighting_res_label = 3
                    elif lighting_value > 90:
                        lighting_res_label = 5

                    return lighting_res_label

                lighting = extract_lighting(fname)
                bias.append(lighting)


                # load image.
                img = Image.open(os.path.join(root, person, fname))
                img = img.convert('L')  # grey image.

                # reduce computation complexity.
                # img = img.resize([s // reduce for s in img.size])

                # convert image to numpy array.
                # img = np.asarray(img).reshape((-1, 1))
                img = np.asarray(img)

                # plt.imshow(img, cmap="gray")
                # plt.title(lighting)
                # plt.show()

                img = img.reshape((img.shape[0], img.shape[1], 1))
                img = np.moveaxis(img, 2, 0)

                # # normalized
                # transform = transforms.Compose([
                #     transforms.ToTensor(),
                #     transforms.Normalize(mean=[0.5],
                #                          std=[0.5])
                # ])
                # transform(img)

                # collect data and label.
                images.append(img)
                labels.append(i)





        # concate all images and labels.
        images = np.array(images, dtype='int64')
        labels = np.array(labels) - 1
        bias = np.array(bias) - 1


        def count(labels):
            for i in range(38):
                c = 0
                for j in range(labels.shape[0]):
                    if labels[j] == i:
                        c = c + 1
                print(str(i) + ': ' + str(c) + '\n')

        # count(labels)

        return images, labels, bias

    def dataset_split(self):
        labels = self.labels
        images = self.images
        bias = self.bias

        training_index = []
        testing_index = []
        training_image, training_label, training_bias = [], [], []
        for i in range(38):
            for lighting in range(5):
                for j in range(labels.shape[0]):
                    if labels[j] == i and bias[j] == lighting:
                        training_index.append(j)
                        break

        for i in range(labels.shape[0]):
            if i not in training_index:
                testing_index.append(i)

        return training_index, testing_index

    def to_tensor(self):
        self.images = torch.tensor(self.images, dtype=torch.float)
        self.labels = torch.tensor(self.labels, dtype=torch.float)
        self.bias = torch.tensor(self.bias, dtype=torch.float)




if __name__ == '__main__':
    load_path = 'raw_data/CroppedYale/'
    dtrain = torch.utils.data.DataLoader(
        YaleBDataset(
            path = load_path,
            mode='train',
        ),
        shuffle=False
    )

    x, y= next(iter(dtrain))
    print(x.shape)
    print(y.shape)

    print('size:', len(dtrain))