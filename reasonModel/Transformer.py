from torch import nn
import torch
import copy
from torch import Tensor as T
import torch.nn.functional as F
import math
######++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, attn_drop: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.head_num = heads
        assert self.d_model % self.head_num == 0
        self.d_k = self.d_model // self.head_num
        self.attn_dropout = nn.Dropout(p=attn_drop)
        self.linears = clones(nn.Linear(self.d_model, self.d_model), 4)
        self.init()

    def init(self):
        for linear in self.linears:
            nn.init.kaiming_uniform_(linear.weight.data)

    def forward(self, query: T, key: T, value: T, mask=None) -> (T, T):
        # print(mask.shape)
        if mask is not None:
            mask = mask.unsqueeze(dim=1)

        batch_size = query.shape[0]
        query, key, value = [l(x).view(batch_size, -1, self.head_num, self.d_k).transpose(1, 2)
                                                      for l, x in zip(self.linears, (query, key, value))]
        x, attention = self_attention(query=query, key=key, value=value, mask=mask, dropout=self.attn_dropout)
        x = x.transpose(1, 2).contiguous() \
            .view(batch_size, -1, self.head_num * self.d_k)
        res = self.linears[-1](x)
        return res, attention

def self_attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    # print('score ', scores.shape)
    # print('mask shape {}'.format(mask.shape))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

######++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.init()

    def init(self):
        nn.init.kaiming_uniform_(self.w_1.weight.data)
        nn.init.kaiming_uniform_(self.w_2.weight.data)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Transformer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, d_model: int, heads: int, attn_drop: float = 0.1, input_drop: float = 0.1):
        super(Transformer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model=d_model, heads=heads, attn_drop=attn_drop)
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, d_ff=4*d_model, dropout=input_drop)
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.ff_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(input_drop)

    def forward(self, query: T, key: T, value: T, x_mask: T = None):
        x_res, _ = self.self_attn.forward(query=query, key=key, value=value, mask=x_mask)
        x_res = x_res + self.dropout(self.self_attn_norm(x_res))
        x_res = x_res + self.dropout(self.ff_norm(self.feed_forward(x_res)))
        return x_res


if __name__ == '__main__':
    x = torch.rand((2, 5, 128))
    # print(x)
    transformer = Transformer(d_model=128, heads=4, attn_drop=0.0, input_drop=0.0)
    x_mask = torch.ones((2,5,5), dtype=torch.bool)
    x_mask[:,:,4] = False
    x_mask[:,4,:] = False
    y = transformer.forward(x, x, x, x_mask)
    print(y)
    print()