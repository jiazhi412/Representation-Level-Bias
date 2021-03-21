import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    input:
        x:
    outputs:
        e1, e2
    """
    def __init__(
            self,
            input_shape=(192, 168, 1),
            e_dim = 100,
            e1_dim=100,
            e2_dim=100,
            nz='tanh'
    ):
        super(Encoder, self).__init__()
        self.input_h, self.input_w, self.input_dep = input_shape
        self.hidden_dim = 16 * self.input_h // 8 * self.input_w // 8
        # self.hidden_dim = 16 * (self.input_h // 32 - 1) * (self.input_w // 32 - 1)

        self.enc_conv1 = nn.Conv2d(self.input_dep, 6, kernel_size=5, stride=2, padding=2)
        self.enc_bn1 = nn.BatchNorm2d(6)

        self.enc_conv2 = nn.Conv2d(6, 6, kernel_size=5, stride=2, padding=2)
        self.enc_bn2 = nn.BatchNorm2d(6)

        self.enc_conv3 = nn.Conv2d(6, 16, kernel_size=5, stride=2, padding=2)
        self.enc_bn3 = nn.BatchNorm2d(16)

        self.enc_conv4 = nn.Conv2d(16, 16, kernel_size=3, stride=2)
        self.enc_bn4 = nn.BatchNorm2d(16)

        self.enc_conv5 = nn.Conv2d(16, 16, kernel_size=3, stride=2)
        self.enc_bn5 = nn.BatchNorm2d(16)

        # self.enc_e = nn.Linear(self.hidden_dim, e_dim)

        self.enc_e1 = nn.Linear(self.hidden_dim, e1_dim)
        self.enc_e2 = nn.Linear(self.hidden_dim, e2_dim)

        if nz == 'tanh':
            self.nz = nn.Tanh()
        elif nz == 'sigmoid':
            self.nz = nn.Sigmoid()
        elif nz == 'relu':
            self.nz = nn.ReLU()

    def forward(self, x):
        x = nn.ReLU()(self.enc_bn1(self.enc_conv1(x)))
        x = nn.ReLU()(self.enc_bn2(self.enc_conv2(x)))
        x = nn.ReLU()(self.enc_bn3(self.enc_conv3(x)))
        # x = nn.ReLU()(self.enc_bn4(self.enc_conv4(x)))
        # x = nn.ReLU()(self.enc_bn5(self.enc_conv5(x)))

        bs, dim, h, w = x.size()
        x = x.view(bs, -1)

        # e = self.nz(self.enc_e(x))

        e1 = self.nz(self.enc_e1(x))
        e2 = self.nz(self.enc_e2(x))
        return e1, e2

if __name__ == '__main__':
    import numpy as np
    enc = Encoder()
    x = torch.randn(1000, 59)
    e1, e2 = enc(x)
    print(e1.size(), e2.size())
    model_parameters = filter(lambda p: p.requires_grad, enc.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    print('Trainable params:', num_params)