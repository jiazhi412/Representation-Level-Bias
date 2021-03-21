import torch.nn as nn
import torch.nn.functional as F
import torch

from model.RLB.RLB_utils import *
from model.MINE.MINE import Mine

class RLB(nn.Module):
    def __init__(self, r_dim, z_dim, opt):
        super().__init__()
        self.batch_size = opt['batch_size']
        # mapping networks (Encoder)
        self.encoder1 = nn.Linear(r_dim, opt['encoder_dim'], bias=False)
        self.encoder2 = nn.Linear(z_dim, opt['encoder_dim'], bias=False)
        # statistics (Mutual information estimation)
        self.MIE = Mine(opt['encoder_dim']*2, hidden_size=opt['hidden_dim'])

    def forward(self, r, z):
        # encoder
        re = F.elu(self.encoder1(r))
        ze = F.elu(self.encoder2(z))
        # select batch
        ze_index, joint_batch = sample_batch_joint((re,ze), self.batch_size)
        marginal_batch = sample_batch_marginal((re,ze), ze_index, self.batch_size)
        # get score
        t = self.MIE(joint_batch)
        et = torch.exp(self.MIE(marginal_batch))
        return t, et
