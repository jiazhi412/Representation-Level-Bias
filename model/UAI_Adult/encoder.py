import torch
import torch.nn as nn

class EncoderGerman(nn.Module):
    """
    input:
        x:
    outputs:
        e1, e2
    """
    def __init__(
        self,
        input_shape=59,
        e1_dim=10,
        e2_dim=20,
        nz='tanh'
    ):
        super(EncoderGerman, self).__init__()
        self.input_shape = input_shape

        self.enc_e1 = nn.Linear(self.input_shape, e1_dim)
        self.enc_e2 = nn.Linear(self.input_shape, e2_dim)

        if nz == 'tanh':
            self.nz = nn.Tanh()
        elif nz == 'sigmoid':
            self.nz = nn.Sigmoid()

    def forward(self, x):
        e1 = self.nz(self.enc_e1(x))
        e2 = self.nz(self.enc_e2(x))
        return e1, e2

if __name__ == '__main__':
    import numpy as np
    enc = EncoderGerman()
    x = torch.randn(1000, 59)
    e1, e2 = enc(x)
    print(e1.size(), e2.size())
    model_parameters = filter(lambda p: p.requires_grad, enc.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    print('Trainable params:', num_params)