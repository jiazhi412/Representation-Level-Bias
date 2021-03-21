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
        drop_rate=0.5,
        output_shape=(192, 168, 1),
        e1_dim=40,
        e2_dim=40,
    ):
        super(Reconstructor, self).__init__()
        self.drop_rate = drop_rate
        self.output_h, self.output_w, self.output_dep = output_shape

        self.trunctor = nn.Dropout(p=drop_rate)

        self.reconst_fc1 = nn.Linear(e1_dim + e2_dim, 16 * self.output_h // 8 * self.output_w // 8)
        self.reconst_bn1 = nn.BatchNorm1d(16 * self.output_h // 8 * self.output_w // 8)

        self.reconst_up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.reconst_conv1 = nn.Conv2d(16, 6, 1)

        self.reconst_up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.reconst_conv2 = nn.Conv2d(6, 6, 1)

        self.reconst_up3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.reconst_conv3 = nn.Conv2d(6, self.output_dep, 1)

    def forward(self, e1, e2):
        e1_truncated = self.trunctor(e1)

        x = torch.cat([e1_truncated, e2], dim=1)
        x = nn.ReLU()(self.reconst_bn1(self.reconst_fc1(x)))

        bs, _ = x.size()
        x = x.view(bs, 16, self.output_h // 8, self.output_w // 8)

        x = self.reconst_up1(x)
        x = nn.Sigmoid()(self.reconst_conv1(x))
        x = self.reconst_up2(x)
        x = nn.Sigmoid()(self.reconst_conv2(x))
        x = self.reconst_up3(x)
        x = nn.Sigmoid()(self.reconst_conv3(x))
        x = x * 255
        return x


if __name__ == '__main__':
    import numpy as np
    e1 = torch.randn(3, 10)
    e2 = torch.randn(3, 20)
    rec = Reconstructor()
    x = rec(e1, e2)
    print(x.shape)
    model_parameters = filter(lambda p: p.requires_grad, rec.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    print('Trainable params:', num_params)