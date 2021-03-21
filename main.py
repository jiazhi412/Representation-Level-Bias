import torch
import numpy as np
import parse_args
import model.MINE as MINE
import model.RLB as RLB
import model.Logits_Loss as LL
import utils

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

opt, model, optim, data = parse_args.collect_args()

# train
result_last = []
if opt['preprocessing'] == 'normal':
    if opt['model'] == 'MINE':
        mi, entropy, bias = MINE.train(data, model, optim, **opt)
    elif opt['model'] == 'RLB':
        mi, entropy, bias = RLB.train(data, model, optim, **opt)
    elif opt['model'] == 'LL':
        Logits_Loss = LL.train(data, model, optim, **opt)
        # print(Logits_Loss[-100:])
    result = (mi, entropy, bias)
    
elif opt['preprocessing'] == 'indices':
    x_section = np.split(data, opt['indices'], axis=1)
    result_ma_indices_list = []
    for i in range(opt['indices']):
        print('indices number: {}'.format(i))
        x = x_section[i]
        mi, entropy, bias = MINE.train(data, model, optim, opt)
        result_ma_indices = MINE.ma(mi, opt['window_size'])
        result_last.append(result_ma_indices[-1])
        result_ma_indices_list.append(result_ma_indices)
    result_ma = np.mean(np.array(result_ma_indices_list), axis=0)

# save result
utils.save_result(opt, result)