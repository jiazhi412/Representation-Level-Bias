import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # original different index
# def sample_batch(data, batch_size=100, sample_mode='joint'):
#     if sample_mode == 'joint':
#         index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
#         batch = data[index]
#     else:
#         joint_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
#         marginal_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
#         batch = np.concatenate([data[joint_index][:,:data.shape[1]//2], data[marginal_index][:,data.shape[1]//2:]], axis=1)
#     return batch

# changed changed index
def sample_batch(data, batch_size=100, sample_mode='joint'):
    x, y = data
    if sample_mode == 'joint':
        index = np.random.choice(range(x.shape[0]), size=batch_size, replace=False)
        batch = np.concatenate([x[index], y[index]], axis = 1)
    elif sample_mode == 'margin':
        joint_index = np.random.choice(range(x.shape[0]), size=batch_size, replace=False)
        marginal_index = np.random.choice(range(y.shape[0]), size=batch_size, replace=False)
        batch = np.concatenate([x[joint_index], y[marginal_index]], axis=1)
    return batch

def sample_batch_joint(data, batch_size=100):
    x, y = data
    index = np.random.choice(range(x.shape[0]), size=batch_size, replace=False)
    batch = torch.cat([x[index], y[index]], dim = 1)
    return index, batch

def sample_batch_marginal(data, y_index, batch_size=100):
    x, y = data
    x_marginal_index = np.random.choice(range(y.shape[0]), size=batch_size, replace=False)
    batch = torch.cat([x[x_marginal_index], y[y_index]], dim=1)
    return batch

def onehot2category(onehot):
    # one hot vector to numerical label
    category = np.zeros((onehot.shape[0],1))
    for i in range(onehot.shape[0]):
        category[i] = np.where(onehot[i,:] == 1)
    category = torch.tensor(category)
    return category

def cal_entropy(y):
    # for category
    if y.size()[1] == 1:
        pass
    # for onehot vector
    else:
        y = onehot2category(y)
    cls_count = torch.stack([y == c for c in y.unique()]).sum(1).float()
    cls_w = cls_count[cls_count > 0] / cls_count.sum()
    entropy = -(cls_w * cls_w.log()).sum().item()
    return entropy

def learn_mine(joint, marginal, mine_net, mine_net_optim, moving_average_et, smoothCoeff=0.01):
    # joint = torch.tensor(joint).float().to(device)
    # marginal = torch.tensor(marginal).float().to(device)
    mine_net.to(device)
    t = mine_net(joint)
    et = torch.exp(mine_net(marginal))
    mi_lowerbound = torch.mean(t) - torch.log(torch.mean(et))
    moving_average_et = (1 - smoothCoeff) * moving_average_et + smoothCoeff * torch.mean(et)

    # unbiasing use moving average
    loss = -(torch.mean(t) - (1 / moving_average_et.mean()).detach() * torch.mean(et))

    # use biased estimator
    # loss = - mi_lowerbound

    mine_net_optim.zero_grad()
    loss.backward()
    mine_net_optim.step()
    return mi_lowerbound, moving_average_et.detach().cpu().numpy(), t.mean().detach().cpu().numpy(), et.mean().detach().cpu().numpy()

# def learn_mine(joint, marginal, mine_net, mine_net_optim, moving_average_et, smoothCoeff=0.01):
#     t = mine_net(joint)
#     et = torch.exp(mine_net(marginal))
#     mi_lowerbound = torch.mean(t) - torch.log(torch.mean(et))
#     moving_average_et = (1 - smoothCoeff) * moving_average_et + smoothCoeff * torch.mean(et)

#     # unbiasing use moving average
#     # loss = -(torch.mean(t) - (1 / moving_average_et.mean()).detach() * torch.mean(et))

#     # use biased estimator
#     loss = - mi_lowerbound

#     mine_net_optim.zero_grad()
#     loss.backward()
#     mine_net_optim.step()
#     return mi_lowerbound, moving_average_et.detach().cpu().numpy(), t.mean().detach().cpu().numpy(), et.mean().detach().cpu().numpy()

# changed
def train(data, mine_net, mine_net_optim, **kwargs):
    ma_et = kwargs['ma_et']
    mi_res = list()
    entropy_res = list()
    bias_res = list()
    iter_num = tqdm(range(kwargs['iter_num']))
    for i in iter_num:
        if kwargs['sample'] == 'same':
            y_index, joint_batch = sample_batch_joint(data, batch_size=kwargs['batch_size'])
            marginal_batch = sample_batch_marginal(data, y_index, batch_size=kwargs['batch_size'])
        elif kwargs['sample'] == 'different':
            joint_batch = sample_batch(data, batch_size=kwargs['batch_size'], sample_mode='joint')
            marginal_batch = sample_batch(data, batch_size=kwargs['batch_size'], sample_mode='margin')
        mi_lb, ma_et, t, et = learn_mine(joint_batch, marginal_batch, mine_net, mine_net_optim, ma_et)
        mi_res.append(mi_lb.detach().cpu().numpy())

        # calculate entropy of y
        # x, y = data
        # sample_y = y[y_index]
        # entropy = cal_entropy(sample_y)
        # entropy_res.append(entropy)
        # bias = mi_res[-1] / entropy_res[-1]
        # bias_res.append(bias)
        entropy_res.append(0)
        bias_res.append(0)
        iter_num.set_postfix({"MI": mi_res[-1],
                              "t": t,
                              "et": et})
                              #"Entropy": entropy_res[-1],
                              #"Bias": bias_res[-1]})

        # save model
        if i % 10000 == 0 and kwargs['save_model'] == True:
            mine_net.save_weights(
                file_path=kwargs['checkpoint_template'].format(i),
                optim=mine_net_optim,
            )
    return mi_res, entropy_res, bias_res

def train_multinormal(data, mine_net, mine_net_optim, **kwargs):
    ma_et = kwargs['ma_et']
    mi_res = list()
    entropy_res = list()
    bias_res = list()
    iter_num = tqdm(range(kwargs['iter_num']))
    for i in iter_num:
        if kwargs['sample'] == 'same':
            y_index, joint_batch = sample_batch_joint(data, batch_size=kwargs['batch_size'])
            marginal_batch = sample_batch_marginal(data, y_index, batch_size=kwargs['batch_size'])
        elif kwargs['sample'] == 'different':
            joint_batch = sample_batch(data, batch_size=kwargs['batch_size'], sample_mode='joint')
            marginal_batch = sample_batch(data, batch_size=kwargs['batch_size'], sample_mode='margin')
        mi_lb, ma_et = learn_mine(joint_batch, marginal_batch, mine_net, mine_net_optim, ma_et)
        mi_res.append(mi_lb.detach().cpu().numpy())

        iter_num.set_postfix({"MI": mi_res[-1]})

        # save model
        if i % 10000 == 0 and kwargs['save_model'] == True:
            mine_net.save_weights(
                file_path=kwargs['checkpoint_template'].format(i),
                optim=mine_net_optim,
            )
    return mi_res

def ma(a, window_size=int(1000)):
    return np.array([np.mean(a[i:i+window_size]) for i in range(0,len(a)-window_size)])

def construct_cov(dim, cc):
    autocov_x1 = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            if i == j:
                autocov_x1[i, j] = 1
    autocov_x2 = autocov_x1

    crosscov = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            if i == j:
                crosscov[i, j] = cc

    tmp1 = np.concatenate((autocov_x1, crosscov), axis=1)
    tmp2 = np.concatenate((crosscov, autocov_x2), axis=1)
    res = np.concatenate((tmp1, tmp2), axis=0)
    # res = np.array([[autocov_x1, crosscov], [crosscov, autocov_x2]]).reshape((dim*2,dim*2))
    return res

if __name__ == '__main__':
    import argparse
    from MINE import Mine
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("running on GPU")
    else:
        device = torch.device("cpu")
        print("running on CPU")

    parser = argparse.ArgumentParser(description='Calculate MINE for multinormal variable')
    parser.add_argument('-s', '--sample-dim', metavar='', type=int, default=20, help='sample dimension')
    opt=vars(parser.parse_args())

    # hyper-parameter
    opt['batch_size'] = 1000
    opt['iter_num'] = 200000
    opt['window_size'] = 1000
    opt['ma_et'] = 1.0
    opt['hidden_size'] = 100

    # pn = np.array([-0.99, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99])
    pn = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99])
    opt['sample_size'] = 3000


    save_dir = os.path.join('./Multinormal_result', str(opt['sample_dim']))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    opt['sample'] = 'same'
    opt['save_model'] = False

    result_dict = dict()
    True_MI = dict()
    for i in range(pn.shape[0]):
        # prepare data
        cc = pn[i]  # correlation coefficient
        cov = construct_cov(opt['sample_dim'], cc)
        det = np.linalg.det(cov)
        True_MI[cc] = -np.log(det) / 2
        data = np.random.multivariate_normal(mean=np.zeros(opt['sample_dim'] * 2), cov=cov, size=opt['sample_size'])
        x = data[:,:data.shape[1]//2]
        y = data[:,data.shape[1]//2:]
        data = (x,y)

        # hyper-parameter
        mine_net = Mine(input_size=opt['sample_dim']*2, hidden_size=opt['hidden_size']).to(device)
        mine_net_optim = optim.Adam(mine_net.parameters(), lr=1e-3)

        # train
        result = train_multinormal(data, mine_net, mine_net_optim, **opt)
        result_ma = ma(result, opt['window_size'])

        # save figure
        plt.plot(range(len(result_ma)), result_ma)
        plt.savefig(os.path.join(save_dir, str(cc) + ".png"))
        # plt.show()

        # save output
        np.savetxt(os.path.join(save_dir, str(cc) + ".txt"), result_ma)

        # save last point mutual information
        result_dict[cc] = result_ma[-1]
        with open(os.path.join(save_dir, str(cc) + "_mean.txt"), 'w') as f:
            f.write(str(result_dict))

    # save final result
    save_path = os.path.join(save_dir, "MI_mean.txt")
    try:
        with open(save_path, 'w') as f:
            f.write(str(result_dict))
        print('Save success!')
    except Exception as e:
        print('Save fail!', e)

    # save true mutual information
    save_path = os.path.join(save_dir, "True_MI.txt")
    with open(save_path, 'w') as f:
        f.write(str(True_MI))










