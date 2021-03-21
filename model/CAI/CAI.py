import torch
import torch.nn as nn
from model.CAI.encoder import Encoder
from model.CAI.reconstructor import Reconstructor
from model.CAI.predictor import Predictor

# train together

class CAI(nn.Module):
    """
    pytorch version of UAI mnist rotation
    """

    def __init__(self, input_shape = (192,168,1), s_dim = 5, h_dim = 100, num_classes = 38):
        super(CAI, self).__init__()
        # ========= param setting ===========
        self.input_shape = input_shape
        self.s_dim = s_dim
        self.h_dim = h_dim
        self.num_classes = num_classes
        # ========= create models ===========
        self.encoder = Encoder(input_shape=self.input_shape, s_dim = self.s_dim, h_dim = self.h_dim)
        self.reconstructor = Reconstructor(output_shape=self.s_dim, h_dim = self.h_dim)
        self.predictor = Predictor(num_classes=self.num_classes)

    def forward(self, x, s, phase=None):
        if phase == 'main':
            h = self.encoder(x, s)
            x_pred = self.predictor(h)
            s_reconst = self.reconstructor(h)
            outputs = [x_pred, s_reconst]
        elif phase == 'for_MINE':
            with torch.no_grad():
                h = self.encoder(x, s)
                h = h.detach()
            outputs = h
        elif phase == None:
            e1, e2 = self.encoder(x)
            x_pred = self.predictor(e1)
            x_reconst = self.reconstructor(e1, e2)
            e1_pred = self.disentangle_2_1(e2)
            e2_pred = self.disentangle_1_2(e1)
            e1_target = self._random_sampling(e1)
            e2_target = self._random_sampling(e2)
            outputs = (x_pred, x_reconst, e1_pred, e2_pred, e1_target, e2_target)
        elif phase == 'test':
            raise NotImplementedError
        else:
            raise RuntimeError('ERROR: phase should be in ["main", "adv", "test"]')

        return outputs

    def _random_sampling(self, x):
        bs, dim = x.size()
        return torch.FloatTensor(bs, dim).uniform_(-1, 1).to(x.device)

    def main_parameters(self):
        res = list(self.encoder.parameters()) + list(self.reconstructor.parameters()) + list(
            self.predictor.parameters())
        return res


    def save_weights(self, file_path, optim_main=None, optim_adv=None):
        torch.save({
            'encoder': self.encoder.state_dict(),
            'reconstructor': self.reconstructor.state_dict(),
            'predictor': self.predictor.state_dict(),
            'optim_main': optim_main.state_dict() if optim_main is not None else None,
        }, file_path)

    def load_weights(self, file_path, optim_main=None, optim_adv=None):
        ckpt = torch.load(file_path)
        self.encoder.load_state_dict(ckpt['encoder'])
        self.reconstructor.load_state_dict(ckpt['reconstructor'])
        self.predictor.load_state_dict(ckpt['predictor'])
        if optim_main is not None:
            optim_main_state_dict = ckpt['optim_main']
            if optim_main_state_dict is None:
                print('WARNING: No optim_main state dict found')
            else:
                optim_main.load_state_dict(optim_main_state_dict)
        if optim_adv is not None:
            optim_adv_state_dict = ckpt['optim_adv']
            if optim_adv_state_dict is None:
                print('WARNING: No optim_adv state dict found')
            else:
                optim_adv.load_state_dict(optim_adv_state_dict)


def test_module():
    x = torch.randn(2, 108).to(device)
    model = UAI().to(device)
    out1 = model(x, 'main') # x_pred, x_reconst, e1_pred, e2_pred, e1_target, e2_target
    for i in out1:
        print(i.shape)
    out2 = model(x, 'adv')
    for i in out2:
        print(i.shape)


def test_module_train():
    # dummy data
    x = torch.randn(2, 108).to(device)
    label = torch.empty(2, 2).random_(2).to(device)
    bias = torch.empty(2, 4).random_(4).to(device)
    # model
    model = UAI().to(device)
    reconst_criterion = nn.MSELoss()
    pred_criterion = nn.MSELoss()
    disentangle_criterion = nn.MSELoss()
    optim_main = torch.optim.Adam(model.main_parameters(), lr=1e-4, weight_decay=1e-4)
    optim_adv = torch.optim.Adam(model.adv_parameters(), lr=1e-3, weight_decay=1e-4)
    # forward main
    optim_main.zero_grad()
    x_pred, x_reconst, e1_pred, e2_pred, e1_target, e2_target = model(x, 'main')
    loss = reconst_criterion(x_reconst, x) + \
           pred_criterion(x_pred, label) + \
           disentangle_criterion(e1_pred, e1_target) + \
           disentangle_criterion(e2_pred, e2_target)
    # loss = pred_criterion(x_pred, label)
    loss.backward()
    optim_main.step()
    # forward adv
    optim_adv.zero_grad()
    e1_pred, e2_pred, e1_target, e2_target = model(x, 'adv')
    loss = disentangle_criterion(e1_pred, e1_target) + disentangle_criterion(e2_pred, e2_target)
    loss.backward()
    optim_adv.step()
    print('=' * 30, 'Pass', '=' * 30)


def test_module_save_and_load():
    # model
    model = UAI().to(device)
    reconst_criterion = nn.MSELoss()
    pred_criterion = nn.CrossEntropyLoss()
    disentangle_criterion = nn.MSELoss()
    optim_main = torch.optim.Adam(model.main_parameters(), lr=1e-4, weight_decay=1e-4)
    optim_adv = torch.optim.Adam(model.adv_parameters(), lr=1e-3, weight_decay=1e-4)

    file_template = 'dummy_model.pth'
    model.save_weights(file_template)
    model.load_weights(file_template)
    print('=' * 30, 'Pass', '=' * 30)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_module()
    test_module_train()
    test_module_save_and_load()