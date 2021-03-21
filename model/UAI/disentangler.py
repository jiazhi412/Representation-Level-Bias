import torch
import torch.nn as nn

class Disentangler(nn.Module):
    """
    input:
        e1 or e2
    output:
        e2 pred or e1 pred
    """
    def __init__(self, in_dim, out_dim, actvation='tanh'):
        super(Disentangler, self).__init__()
        self.disentangle_fc1 = nn.Linear(in_dim, out_dim)
        if actvation == 'tanh':
            self.nz = nn.Tanh()
        elif actvation == 'sigmoid':
            self.nz = nn.Sigmoid()

    def forward(self, x):
        x = self.nz(self.disentangle_fc1(x))
        return x