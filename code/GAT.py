import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.6):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.a = nn.Parameter(torch.FloatTensor(2 * out_features, 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data)
        nn.init.xavier_uniform_(self.a.data)

    def forward(self, x, adj):
        h = torch.mm(x, self.weight)
        N = h.size()[0]  # 获取行数，h.size()[1]获取列数

        # repeat(m,n): 行数乘m，列数乘n
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = F.mish(torch.matmul(a_input, self.a).squeeze(2))  # [N,N]

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # 如果邻接矩阵中的元素大于零，则将对应位置的注意力权重设置为e,否则将其设置为zero_vec。
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        return F.mish(h_prime)


class GAT(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, dropout, n_heads):
        super(GAT, self).__init__()
        attentions = [GraphAttentionLayer(n_feat, n_hid, dropout) for _ in range(n_heads)]
        self.attentions = nn.Sequential(*attentions)
        self.out_att = GraphAttentionLayer(n_hid * n_heads, n_class, dropout)

    def forward(self, x, adj):
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        return self.out_att(x, adj)

