from math import sqrt
import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim_in, dim_k, dim_v, num_heads, drop_rate):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k // num_heads)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        x = x.unsqueeze(1)
        # x: tensor of shape (batch, n, dim_in)
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        nh = self.num_heads
        dk = self.dim_k // nh  # dim_k of each head
        dv = self.dim_v // nh  # dim_v of each head

        q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        v = self.linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)  # (batch, nh, n, dv)

        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact  # batch, nh, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, nh, n, n

        att = torch.matmul(dist, v)  # batch, nh, n, dv
        att = self.dropout(att)
        att = att.transpose(1, 2).reshape(batch, n, self.dim_v)  # batch, n, dim_v
        att = att.squeeze(1)
        # x = self.linear_v(x).squeeze(1)
        return att


# 跨多头自注意力模型
class CrossAttention(nn.Module):
    def __init__(self, dim_in, dim_k, dim_v, num_heads, drop_rate):
        super(CrossAttention, self).__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k // num_heads)
        # self.layer_norm = nn.LayerNorm(dim_v)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x, s):
        s = torch.matmul(s, x).unsqueeze(1)
        # x: tensor of shape (batch, n, dim_in)
        batch, n, dim_in = s.shape
        assert dim_in == self.dim_in

        nh = self.num_heads
        dk = self.dim_k // nh  # dim_k of each head
        dv = self.dim_v // nh  # dim_v of each head

        q = self.linear_q(s).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        v = self.linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)  # (batch, nh, n, dv)

        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact  # batch, nh, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, nh, n, n

        att = torch.matmul(dist, v)  # batch, nh, n, dv
        att = self.dropout(att)
        att = att.transpose(1, 2).reshape(batch, n, self.dim_v)  # batch, n, dim_v
        att = att.squeeze(1)
        # att = self.layer_norm(att + self.linear_v(x))
        # x = self.linear_v(x).squeeze(1)
        return att


class SA_Fusion(nn.Module):
    def __init__(self, num, dim_in, dim_k, dim_v, num_heads, drop_rate):
        super(SA_Fusion, self).__init__()
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.dropout = nn.Dropout(drop_rate)
        self.sa = MultiHeadSelfAttention(dim_in, dim_k, dim_v, num_heads, drop_rate)
        self.fusion = nn.Linear(dim_v, dim_v)
        self.a = nn.Parameter(torch.rand(num))
        # encoder = []
        # for i in range(num):
        #     encoder.append(nn.Linear(dim_in[i], dim_k*2))
        # self.encoders = nn.Sequential(*encoder)

    def forward(self, x_list):
        a = []
        for i, x in enumerate(x_list):
            a.append(self.a[i] * x)
        x = torch.cat([b for b in a], dim=1)
        # x = self.a[0] * self.encoders[0](x_list[0])
        # for i in range(1, len(x_list)):
        #     x += self.a[i] * self.encoders[i](x_list[i])
        x = self.sa(x)
        x = self.fusion(x)
        x = self.dropout(x)
        return x
