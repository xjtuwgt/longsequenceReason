from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_input, d_mid, d_out, dropout=0.1):
        super(MLP, self).__init__()
        self.w_1 = nn.Linear(d_input, d_mid)
        self.w_2 = nn.Linear(d_mid, d_out)
        self.dropout = nn.Dropout(dropout)
        self.init()

    def init(self):
        nn.init.kaiming_uniform_(self.w_1.weight.data)
        nn.init.kaiming_uniform_(self.w_2.weight.data)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))