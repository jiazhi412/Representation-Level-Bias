import torch
import torch.nn as nn

class DisentanglerGerman(nn.Module):
    """
    input:
        e1 or e2
    output:
        e2 pred or e1 pred
    """
    def __init__(self, in_dim, out_dim, actvation='tanh'):
        super(DisentanglerGerman, self).__init__()
        self.disentangle_fc1 = nn.Linear(in_dim, out_dim)
        if actvation == 'tanh':
            self.nz = nn.Tanh()
        elif actvation == 'sigmoid':
            self.nz = nn.Sigmoid()
        else:
            raise NotImplementedError('ERROR: Unknown activation {}'.format(actvation))

    def forward(self, x):
        x = self.nz(self.disentangle_fc1(x))
        return x