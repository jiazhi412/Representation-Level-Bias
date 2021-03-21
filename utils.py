import numpy as np
import pickle
import glob
import parse
from sklearn import preprocessing
import utils
import os
import torch
import matplotlib.pyplot as plt
from collections import Counter

import model.MINE as MINE
import model.RLB as RLB
from model.UAI.UAI import UAI as UAI_YaleB
from model.CAI.CAI import CAI
from model.UAI_Adult.UAI import UAI as UAI_Adult
from data_loader.YaleB import YaleBDataset
from data_loader.Adult import AdultDataset

# 1. running
def set_random_seed(seed_number):
    torch.manual_seed(seed_number)
    np.random.seed(seed_number)

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_pkl(pkl_data, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(pkl_data, f)

def load_pkl(load_path):
    with open(load_path, 'rb') as f:
        pkl_data = pickle.load(f)
    return pkl_data

def save_result(opt, result):
    mi, entropy, bias = result
    mi_ma, entropy_ma, bias_ma = RLB.ma(mi, opt['window_size']), MINE.ma(entropy, opt['window_size']), MINE.ma(bias, opt['window_size'])
    result_last = { "Max" : bias_ma.max(), 
                    "Min" : bias_ma.min(),
                    "Mean" : bias_ma.mean(),
                    "Last mi" : mi_ma[-1], 
                    "Last entropy" : entropy_ma[-1], 
                    "Last bias" : bias_ma[-1]}

    # save figure
    if opt['experiment'].startswith('colored'):
        bias_ma=np.array(bias_ma)
        np.save(opt['save_path'] + '.npy', bias_ma)
        plt.plot(range(bias_ma.shape[0]), bias_ma)
        plt.savefig(opt['save_path'] + "_bias.png")
    else:
        plt.plot(range(mi_ma.shape[0]), mi_ma)
        plt.savefig(opt['save_path'] + "_mi.png")
        # plt.show()

        plt.plot(range(entropy_ma.shape[0]), entropy_ma)
        plt.savefig(opt['save_path'] + "_entropy.png")
        # plt.show()

        plt.plot(range(bias_ma.shape[0]), bias_ma)
        plt.savefig(opt['save_path'] + "_bias.png")
        # plt.show()

    try:
        with open((opt['save_path'] + ".txt"), 'w') as f:
            f.write("Parameters: \n")
            for k,v in opt.items():
                f.write('   ' + str(k) + ': ' + str(v) + '\n')
            f.write("\nResult: \n")
            for k,v in result_last.items():
                f.write('   ' + str(k) + ': ' + str(v) + '\n')
        print('Save success!')
    except Exception as e:
        print('Save fail!', e)
    f.close()

# 2. experiment
def onehot2category(onehot):
    # one hot vector to numerical label
    category = np.zeros((onehot.shape[0],1))
    for i in range(onehot.shape[0]):
        category[i] = np.where(onehot[i,:] == 1)
    category = torch.tensor(category)
    return category

def category2onehot(category):
    encoder = preprocessing.OneHotEncoder()
    encoder.fit(category)
    newCol = []
    for c in category:
        c_array = np.array(c).reshape(-1, 1)
        newCol.append(encoder.transform(c_array).toarray())
    onehot = np.array(newCol).reshape(category.shape[0],-1)
    onehot = torch.tensor(onehot)
    return onehot

def normalized(input):
    output = torch.zeros(input.size())
    for ind, ele in enumerate(input):
        if ele.max() - ele.min() != 0:
            output[ind] = (ele - ele.min()) / (ele.max() - ele.min())
    return output

def filtered_zeros(features, labels):
    features_res = []
    labels_res = []
    first = True
    for i in range(labels.size()[0]):
        # if torch.count_nonzero(features[i]) == 0 or torch.count_nonzero(labels[i]) == 0:
        if torch.count_nonzero(features[i]) == 0:
            pass
        elif first:
            features_res = features[i].view((1,-1))
            labels_res = labels[i].view((1,-1))
            first = False
        else:
            features_res = torch.cat((features_res, features[i].view((1,-1))), dim = 0)
            labels_res = torch.cat((labels_res, labels[i].view((1,-1))), dim=0)
    return features_res, labels_res

def count(input,opt):
    distribution = Counter(input)
    print(distribution)
    if opt['experiment'].startswith('FairFace') and opt['bias'] == 'race':
        opt['distribution'] = np.array(list(distribution.values()))
    elif opt['experiment'].startswith('celeba'):
        opt['distribution'] = np.array(list(distribution.values()))

# 3. Specialized load function
def load_model(dataset_name, model_name):
    if model_name == 'UAI':
        if dataset_name == 'YaleB':
            model = UAI_YaleB()
        elif dataset_name == 'Adult':
            model = UAI_Adult()
        checkpoint_template = os.path.join('checkpoint', dataset_name, 'UAI_epoch_{}.pth')
    elif model_name == 'CAI':
        model = CAI()
        checkpoint_template = os.path.join('checkpoint', dataset_name, 'CAI_epoch_{}.pth')
    ckpt_files = sorted(glob.glob(checkpoint_template.format('*')),
                        key=lambda x: int(parse.parse(checkpoint_template, x)[0]))
    if len(ckpt_files) > 0:
        ckpt = ckpt_files[-1]
        print('INFO: Resume from {}'.format(ckpt))
        model.load_weights(ckpt)
        init_epoch = int(parse.parse(checkpoint_template, ckpt)[0])  # get init epoch
    return model

def load_data(dataset_name = 'YaleB', bias_name = 'gender'):
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    if dataset_name == 'YaleB':
        load_path = 'data_loader/data/CroppedYale/'
        dataset = torch.utils.data.DataLoader(
            YaleBDataset(
                path=load_path,
                mode='all'
            ),
            batch_size=2414,
            shuffle=True,
            **kwargs
        )
    elif dataset_name == 'Adult':
        load_path = "./data_loader/data/Adult/newData.csv"
        dataset = torch.utils.data.DataLoader(
            AdultDataset(
                path=load_path,
                mode='all',
                quick_load=True,
                bias_name=bias_name
            ),
            batch_size=48842,
            shuffle=True,
            **kwargs
        )
    return dataset

# for UAI and CAI
def get_r_z(model_name, dataset_name, feature_name, bias_name='label', onehot_or_category='onehot'):
    # load de-biasing model and dataset
    model = load_model(dataset_name, model_name)
    dataset = load_data(dataset_name, bias_name)

    # load data to get bias or label
    x, label, bias = next(iter(dataset))
    label = label.reshape((-1,1))
    if bias_name == 'label':
        y = label
    else: #bias_name == 'bias':
        y = bias
        y = onehot2category(y)
    if onehot_or_category == 'onehot':
        y = category2onehot(y)
    elif onehot_or_category == 'category':
        pass

    # rerun model to get feature
    if model_name == 'UAI':
        e1, e2 = model(x, 'for_MINE')
        if feature_name == 'e1':
            feature = e1
        elif feature_name == 'e2':
            feature = e2
        else:
            raise RuntimeError('ERROR: for UAI, feature_name should be in ["e1", "e2"]')
    elif model_name == 'CAI':
        h = model(x, bias.float(), 'for_MINE')
        if feature_name == 'h':
            feature = h
        else:
            raise RuntimeError('ERROR: for CAI, feature_name should be in ["h"]')
    return feature, y

def load_coloredMNIST(colored_std, experiment, label_or_bias = 'label', onehot_or_category='onehot'):
    # representation-level bias for input data
    if experiment == 'colored_data':
        path = 'data_loader/data/ColoredMNIST/feature_data_{}.pkl'
    # representation-level bias for trained model
    elif experiment == 'colored_model':
        path = 'data_loader/data/ColoredMNIST/feature_model_{}.pkl'
    file = utils.load_pkl(path.format(colored_std))
    features = file['feature']
    labels = file['label']
    colors = file['color']

    # data preprocessing
    y = None
    if label_or_bias == 'label':
        y = labels
        y = y.view(-1, 1)
    elif label_or_bias == 'color':
        y = colors
    features, y = filtered_zeros(features, y)
    features_res = normalized(features)
    return features_res, y

def load_FairFace(opt, experiment, percentage, protected_attribute, onehot_or_category='onehot'):
    dir = 'data_loader/data/FairFace' 
    if opt['more_race']:
        race_num = '7'
    else:
        race_num = '4'
    mode = experiment.split("_")[-1]
    # imbalance by percentage (gender or race)
    if experiment == 'FairFace_attr_baseline_model':
        path = f'{dir}/{protected_attribute}_{race_num}_attr/feature_{mode}_{percentage}.pkl'
    # debiasing model comparison
    elif model_name == 'FairFace_baseline':
        test_result = utils.load_pkl(os.path.join(dir, 'test_baseline.pkl'))
    elif model_name == 'FairFace_domain_independent':
        test_result = utils.load_pkl(os.path.join(dir, 'test_domain_independent.pkl'))
    elif model_name == 'FairFace_weighting':
        test_result = utils.load_pkl(os.path.join(dir, 'test_weighting.pkl'))
    elif model_name == 'FairFace_domain_adaption':
        test_result = utils.load_pkl(os.path.join(dir, 'test_domain_discriminative.pkl'))
    elif model_name == 'FairFace_representation_disentanglement_uniconf':
        test_result = utils.load_pkl(os.path.join(dir, 'test_uniconf_adv.pkl'))
    elif model_name == 'FairFace_representation_disentanglement_gradproj':
        test_result = utils.load_pkl(os.path.join(dir, 'test_gradproj_adv.pkl'))
    file = utils.load_pkl(path)
    features = file['feature']
    y = file[protected_attribute][:features.shape[0]]
    count(y, opt)

    # data preprocessing
    y = np.array(y).reshape((-1,1))
    y = category2onehot(y)
    if onehot_or_category == 'onehot':
        pass
    elif onehot_or_category == 'category':
        y = onehot2category(y)    
    features = torch.tensor(features)
    # y = torch.tensor(y)
    features, y = filtered_zeros(features, y)
    features_res = normalized(features)
    return features_res, y

def load_StyleGan2(opt, experiment, percentage, protected_attribute, onehot_or_category='onehot'):
    dir = 'data_loader/data/StyleGan2' 
    mode = experiment.split("_")[-1]
    # imbalance by entropy (gender or race)
    path = f'{dir}/{protected_attribute}_attr/feature_{mode}_{percentage}.pkl'
    file = utils.load_pkl(path)
    features = file['feature']
    y = file[protected_attribute][:features.shape[0]]
    count(y, opt)

    # data preprocessing
    y = np.array(y).reshape((-1,1))
    y = category2onehot(y)
    if onehot_or_category == 'onehot':
        pass
    elif onehot_or_category == 'category':
        y = onehot2category(y)    
    features = torch.tensor(features)
    # y = torch.tensor(y)
    features, y = filtered_zeros(features, y)
    features_res = normalized(features)
    return features_res, y

def load_celeba(opt, model_name, label_or_bias = 'label', onehot_or_category='onehot'):
    # get feature
    path = 'data_loader/data/Celeba/'
    # debiasing model comparison
    if model_name == 'celeba_baseline':
        test_result = utils.load_pkl(os.path.join(path, 'test_baseline.pkl'))
    elif model_name == 'celeba_domain_independent':
        test_result = utils.load_pkl(os.path.join(path, 'test_domain_independent.pkl'))
    elif model_name == 'celeba_weighting':
        test_result = utils.load_pkl(os.path.join(path, 'test_weighting.pkl'))
    elif model_name == 'celeba_domain_adaption':
        test_result = utils.load_pkl(os.path.join(path, 'test_domain_discriminative.pkl'))
    elif model_name == 'celeba_representation_disentanglement_uniconf':
        test_result = utils.load_pkl(os.path.join(path, 'test_uniconf_adv.pkl'))
    elif model_name == 'celeba_representation_disentanglement_gradproj':
        test_result = utils.load_pkl(os.path.join(path, 'test_gradproj_adv.pkl'))
    feature = test_result['feature']
    feature = torch.tensor(feature)

    # get label and bias
    target_dict = utils.load_pkl(os.path.join(path, 'labels_dict'))
    test_key_list = utils.load_pkl(os.path.join(path, 'test_key_list'))
    targets = []
    for key in test_key_list:
        targets.append(target_dict[key])
    targets = torch.tensor(targets)
    label = targets[:, :39]
    gender = targets[:, 39]
    count(gender.tolist(), opt)

    # data preprocessing
    y = None
    if label_or_bias == 'label':
        y = label
    elif label_or_bias == 'gender':
        y = gender
        y = y.view(-1,1)
    if onehot_or_category == 'onehot':
        y = category2onehot(y)
    elif onehot_or_category == 'category':
        pass
    feature_res = normalized(feature)
    return feature_res, y

# 4. synthesized representations
def attack_generator(attack, r, z):
    if attack == 'shuffleR':
        r = utils.random_shuffle(r)
    # attack 2 (fake R)
    elif attack == 'fakeR':
        r = torch.randn(r.size())
    # attack 3 (shuffle Z)
    elif attack == 'shuffleZ':
        z = utils.random_shuffle(z)
    # attack 4 (fake Z)
    elif attack == 'fakeZ':
        class_num = 2
        z = utils.fake_generator(class_num, z.size()[0]) 

    # # attack 4
    # x, y = data
    # y = utils.switch(y)
    # data = (x,y)

    # # attack 5
    # r, z = data
    # z = utils.change_gender(z)
    # data = (r,z)
    return r, z

def change_gender(gender):
    res = torch.zeros(gender.size())
    for i in range(gender.size()[0]):
        if gender[i] == 0:
            res[i] = -1
        else:
            res[i] = 1
    return res

def fake_generator(class_num, sample_num):
    import itertools
    c = itertools.cycle(range(class_num))
    category = []
    for i in range(sample_num):
        category.append(next(c))
    category = np.array(category).reshape((-1,1))
    onehot = category2onehot(category)
    return onehot

def random_shuffle(y):
    # With view
    idx = torch.randperm(y.size()[0])
    res = y[idx]
    return res

def switch(y):
    # output, inverse_indices = torch.unique(y, sorted=True, return_inverse=True)
    for idx, ele in enumerate(y):
        if (ele == torch.tensor([1,0], dtype=torch.float64)).all():
            y[idx] = torch.tensor([0,1], dtype=torch.float64)
        else:
            y[idx] = torch.tensor([1,0], dtype=torch.float64)
    return y

def replace_zero(r):
    for i in range(r.size()[0]):
        for j in range(r.size()[1]):
            if r[i][j] == 0:
                r[i][j] = torch.rand(1)
    return r