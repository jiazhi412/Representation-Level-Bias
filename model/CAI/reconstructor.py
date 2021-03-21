'''
Corresponding to decoder in UAI
'''
import torch
import torch.nn as nn

class Reconstructor(nn.Module):
    """
    inputs:
        e1, e2
    """
    def __init__(
        self,
        output_shape=5,
        h_dim=40,
        hidden_dim=40,
        nz='tanh'
    ):
        super(Reconstructor, self).__init__()
        self.dec1 = nn.Linear(h_dim, hidden_dim)
        self.dec_bn1 = nn.BatchNorm1d(hidden_dim)
        self.dec2 = nn.Linear(hidden_dim, output_shape)
        self.dec_bn2 = nn.BatchNorm1d(output_shape)

        if nz == 'tanh':
            self.nz = nn.Tanh()
        elif nz == 'sigmoid':
            self.nz = nn.Sigmoid()


    def forward(self, h):
        s = self.nz(self.dec_bn1(self.dec1(h)))
        s = self.nz(self.dec_bn2(self.dec2(s)))

        return s


    # def __init__(
    #     self,
    #     output_shape=5,
    #     h_dim=40,
    #     hidden_dim=40,
    #     nz='tanh'
    # ):
    #     super(Reconstructor, self).__init__()
    #     self.dec1 = nn.Linear(h_dim, hidden_dim)
    #     self.dec2 = nn.Linear(hidden_dim, output_shape)
    #
    #     if nz == 'tanh':
    #         self.nz = nn.Tanh()
    #     elif nz == 'sigmoid':
    #         self.nz = nn.Sigmoid()
    #
    #
    # def forward(self, h):
    #     s = self.nz(self.dec1(h))
    #     s = self.nz(self.dec2(s))
    #
    #     return s




if __name__ == '__main__':
    import numpy as np
    e1 = torch.randn(3, 10)
    e2 = torch.randn(3, 20)
    rec = ReconstructorGerman()
    x = rec(e1, e2)
    print(x.shape)
    model_parameters = filter(lambda p: p.requires_grad, rec.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    print('Trainable params:', num_params)