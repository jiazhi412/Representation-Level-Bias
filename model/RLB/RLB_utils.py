import numpy as np
from tqdm import tqdm
import torch

def sample_batch_joint(data, batch_size=100):
    x, y = data
    index = np.random.choice(range(x.shape[0]), size=batch_size, replace=False)
    batch = torch.cat([x[index], y[index]], dim=1)
    return index, batch

def sample_batch_marginal(data, y_index, batch_size=100):
    x, y = data
    x_marginal_index = np.random.choice(range(y.shape[0]), size=batch_size, replace=False)
    batch = torch.cat([x[x_marginal_index], y[y_index]], dim=1)
    return batch

def learn_MI(r, z, net, optim, moving_average_et, smoothCoeff=0.01):
    t, et = net(r, z)
    mi_lowerbound = torch.mean(t) - torch.log(torch.mean(et))
    if moving_average_et == None:
        # first period: using moving average to replace exponential moving average
        moving_average_et = torch.mean(et)
    else:
        moving_average_et = (1 - smoothCoeff) * moving_average_et + smoothCoeff * torch.mean(et)

    # unbiasing use moving average
    loss = -(torch.mean(t) - (1 / moving_average_et.mean()).detach() * torch.mean(et))

    # use biased estimator
    # loss = - mi_lb

    optim.zero_grad()
    loss.backward()
    optim.step()
    return mi_lowerbound, moving_average_et.detach().cpu().numpy()

def train(data, net, optim, **kwargs):
    r, z = data
    net.to(kwargs['device'])
    ma_et = kwargs['ma_et']
    mi_list = list()
    entropy_list = list()
    bias_list = list()
    iter_num = tqdm(range(kwargs['iter_num']))
    for i in iter_num:
        # estimate mutual informance
        mi_lb, ma_et = learn_MI(r, z, net, optim, ma_et, smoothCoeff = 2 / (r.size()[0]/kwargs['batch_size']+1))
        mi_list.append(mi_lb.detach().cpu().numpy())

        # calculate entropy
        if kwargs['experiment'].startswith("FairFace"):
            if kwargs['bias'] == 'gender':
                entropy = cal_entropy(kwargs['percentage'])
            elif kwargs['bias'] == 'race':
                entropy = cal_entropy_list(kwargs['distribution'])
        elif kwargs['experiment'].startswith("celeba"):
            entropy = cal_entropy_list(kwargs['distribution'])
        elif kwargs['experiment'].startswith("colored"):
            # no experiment entropy
            entropy = 1
        elif kwargs['experiment'].startswith("StyleGan2"):
            entropy = kwargs['entropy']          
        entropy_list.append(entropy)
        bias_list.append(mi_list[-1] / entropy_list[-1])

        # show off
        iter_num.set_postfix({"MI": mi_list[-1],
                                "Entropy": entropy_list[-1],
                                "Bias": bias_list[-1]})
      
        # save model
        if i % 10000 == 0 and kwargs['save_model'] == True:
            net.save_weights(
                file_path=kwargs['checkpoint_template'].format(i),
                optim=optim,
            )
        # judge if stop before the max iter number because of the convergence
        if kwargs['converge_criterion'] and kwargs['stop_region'] < i and is_converge(mi_list, kwargs):
            break
    return mi_list, entropy_list, bias_list

def cal_entropy(ratio):
    input = np.array([ratio, 1-ratio])
    s = input.sum()
    res = 0
    for i in input:
        res += -(i/s) * np.log(i/s)
    return res

def cal_entropy_list(input):
    s = input.sum()
    res = 0
    for i in input:
        res += -(i/s) * np.log(i/s)
    return res

def is_converge(mi_list, opt):
    flag = False
    stop_error = opt['stop_error']
    window_size = opt['stop_region']
    region1 = np.array([mi_list[i] for i in range(-window_size//2,0)])
    region2 = np.array([mi_list[i] for i in range(-window_size, -window_size//2)])
    if abs(region1.mean() - region2.mean()) <= stop_error or region1.mean() < region2.mean():
        flag = True
    return flag

def ma(a, window_size=int(1000)):
    return np.array([np.mean(a[i:i+window_size]) for i in range(0,len(a)-window_size)])
