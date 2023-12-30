import torch
from torch import nn
from math import sqrt


def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


class Self_Attention(nn.Module):
    # input : seq_len * input_dim
    # q : input_dim * dim_k
    # k : input_dim * dim_k
    # v : input_dim * dim_v
    # q, k have same dimension
    def __init__(self, input_dim, dim_k, dim_v):
        super(Self_Attention, self).__init__()
        self.q = nn.Linear(input_dim, dim_k)
        self.k = nn.Linear(input_dim, dim_k)
        self.v = nn.Linear(input_dim, dim_v)
        self._norm_fact = 1 / sqrt(dim_k)

    def forward(self, x):
        Q = self.q(x)  # Q: seq_len * dim_k
        K = self.k(x)  # K: seq_len * dim_k
        V = self.v(x)  # V: seq_len * dim_v

        atten = nn.Softmax(dim=1)(
            torch.mm(Q, K.T)) * self._norm_fact  # softmax使每一行和为1

        output = torch.mm(atten, V)  # Q * K.T() * V # batch_size * seq_len * dim_v

        return output


class Self_Attention_Muti_Head(nn.Module):
    # input : seq_len * input_dim
    # q : input_dim * dim_k
    # k : input_dim * dim_k
    # v : input_dim * dim_v
    # q, k have same dimension
    def __init__(self, input_dim, dim_k, dim_v, nums_head):
        super(Self_Attention_Muti_Head, self).__init__()
        assert dim_k % nums_head == 0
        assert dim_v % nums_head == 0
        self.nums_head = nums_head
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.input_dim = input_dim
        self.model = Self_Attention(self.input_dim, self.dim_k, self.dim_v)
        self.model.apply(xavier_init)

    def forward(self, x):
        SA = []
        for i in range(self.nums_head):
            SA.append(self.model(x))
        output = SA[0]
        for i in range(1, len(SA)):
            output = torch.cat((output, SA[i]), 1)
        return output
