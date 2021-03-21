import torch
import torch.nn as nn

class PredictorGerman(nn.Module):
    """
    input:
        e1
    output:
        some prediction target
    """
    def __init__(self, e1_dim=10, num_classes=2):
        super(PredictorGerman, self).__init__()
        self.pred_bn1 = nn.BatchNorm1d(e1_dim)
        self.pred_fc1 = nn.Linear(e1_dim, 10)
        self.pred_bn2 = nn.BatchNorm1d(10)
        self.pred_fc2 = nn.Linear(10, num_classes)
        self.pred_bn3 = nn.BatchNorm1d(num_classes)

    def forward(self, x):
        x = self.pred_bn1(x)
        x = nn.ReLU(True)(self.pred_bn2(self.pred_fc1(x)))
        x = nn.ReLU(True)(self.pred_bn3(self.pred_fc2(x)))

        # x = nn.ReLU(True)(self.pred_fc1(x))
        # x = nn.ReLU(True)(self.pred_fc2(x))
        return x

if __name__ == '__main__':
    import numpy as np
    predictor = PredictorGerman()
    x = torch.randn(2, 10)
    pred = predictor(x)
    print("pred is ", pred)
    print(pred.shape)
    model_parameters = filter(lambda p: p.requires_grad, predictor.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    print('Trainable params:', num_params)