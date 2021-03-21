import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

def learn_LL(r, z, net, optim):
    optim.zero_grad()
    z_logits = net(r)
    # print(z_logits.size())
    loss = F.cross_entropy(z_logits, z)
    _, predicted = torch.max(z_logits.data, 1)
    total = z.size(0)
    correct = (predicted == z).sum().item()
    # print('Accuracy: %d %%' % (
    # 100 * correct / total))
    acc = 100 * correct / total
    loss.backward()
    optim.step()
    return z_logits, loss, acc

def train(data, net, optim, **kwargs):
    r, z = data
    # print(z)
    # print(1)
    # print(z.cpu().numpy())
    # z = onehot2category(z.cpu().numpy())
    if kwargs['attack'] == 'fakeZ':
        z = onehot2category(z.cpu().numpy())
    # print(z.size())
    z = z.type(torch.LongTensor).view(z.size()[0]).to(kwargs['device'])
    # print(r.size())
    # print(z.size())
    net.to(kwargs['device'])
    loss_list = list()
    output_list = []
    iter_num = tqdm(range(kwargs['iter_num']))
    for i in iter_num:
        z_logits, loss, train_mAP = learn_LL(r, z, net, optim)
        loss_list.append(loss.detach().item())
        train_predict_prob = inference(z_logits)
        # train_mAP = average_precision_score(z.cpu(), train_predict_prob)
        iter_num.set_postfix({"Logits Loss": loss_list[-1],
                              "Acc": train_mAP})

        # output_list.append(z_logits)
      
        # save model
        if i % 10000 == 0 and kwargs['save_model'] == True:
            net.save_weights(
                file_path=kwargs['checkpoint_template'].format(i),
                optim=optim,
            )
    # train_output = torch.cat(output_list)
    # train_predict_prob = inference(train_output)
   
    print('train mAP: {}'.format(train_mAP))
        # judge if stop before the max iter number because of the convergence
        # if kwargs['stop_region'] < i and is_converge(loss_list, kwargs):
        #     break
    return loss_list

def inference(output):
        predict_prob = torch.sigmoid(output)
        return predict_prob.cpu().detach().numpy()

def is_converge(mi_res, opt):
    flag = False
    stop_error = opt['stop_error']
    window_size = opt['stop_region']
    region1 = np.array([mi_res[i] for i in range(-window_size//2,0)])
    region2 = np.array([mi_res[i] for i in range(-window_size, -window_size//2)])
    if abs(region1.mean() - region2.mean()) <= stop_error or region1.mean() < region2.mean():
        flag = True
    return flag

def onehot2category(onehot):
    # one hot vector to numerical label
    category = np.zeros((onehot.shape[0],1))
    for i in range(onehot.shape[0]):
        category[i] = np.where(onehot[i,:] == 1)
    category = torch.tensor(category)
    return category
