import torch.nn as nn
import torch.nn.functional as F

class LL(nn.Module):
    def __init__(self, input_size=100, class_num=2, hidden_size=100):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, class_num)
        # self.fc = nn.Linear(input_size, class_num)
        self._init_weights()

    def forward(self, input):
        output = F.elu(self.fc1(input))
        output = F.elu(self.fc2(output))
        output = self.fc3(output)
        # output = F.elu(self.fc(input))
        return output

    def _init_weights(self):
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc3.weight, std=0.02)
        nn.init.constant_(self.fc3.bias, 0)
        
        # nn.init.normal_(self.fc.weight, std=0.02)
        # nn.init.constant_(self.fc.bias, 0)