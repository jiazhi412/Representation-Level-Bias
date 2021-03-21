'''
Corresponding to decoder in UAI
'''
import torch
import torch.nn as nn

class ReconstructorGerman(nn.Module):
    """
    inputs:
        e1, e2
    """
    def __init__(
        self,
        drop_rate=0.5,
        output_shape=59,
        e1_dim=10,
        e2_dim=20,
    ):
        super(ReconstructorGerman, self).__init__()
        self.drop_rate = drop_rate
        self.output_shape = output_shape

        self.trunctor = nn.Dropout(p=drop_rate)

        self.reconst_fc1 = nn.Linear(e1_dim+e2_dim, self.output_shape)
        # self.reconst_bn1 = nn.BatchNorm1d(self.output_shape)

    def forward(self, e1, e2):
        e1_truncated = self.trunctor(e1)
        x = torch.cat([e1_truncated, e2], dim=1)
        # x = nn.ReLU(True)(self.reconst_bn1(self.reconst_fc1(x)))

        x = nn.ReLU(True)(self.reconst_fc1(x))
        return x


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