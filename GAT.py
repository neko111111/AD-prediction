import torch
from torch import nn
from torch.nn import functional as f


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat
        self.alpha = alpha

        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.a = nn.Parameter(torch.FloatTensor(2*out_features, 1))
        nn.init.xavier_normal_(self.weight.data)
        nn.init.xavier_normal_(self.a.data)

        self.LeakyReLU = nn.LeakyReLU(self.alpha)

    def forward(self, x, adj):
        Wh = torch.mm(x, self.weight)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.LeakyReLU(torch.matmul(a_input, self.a).squeeze(2))
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = f.softmax(attention, dim=1)
        attention = f.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return f.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]
        Wh_repeat_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeat_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeat_in_chunks, Wh_repeat_alternating], dim=1)
        return all_combinations_matrix.view(N, N, 2*self.out_features)


class GAT(nn.Module):
    def __init__(self, num_features, num_hiddens, num_classes, dropout, alpha, num_heads):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attention_heads = nn.ModuleList([
            GraphAttentionLayer(num_features, num_hiddens, dropout=dropout, alpha=alpha, concat=True) for _ in range(num_heads)
        ])
        self.out_layer = GraphAttentionLayer(num_hiddens * num_heads, num_classes, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = f.dropout(x, self.dropout, training=self.training)
        x = torch.cat([attn_head(x, adj) for attn_head in self.attention_heads], dim=1)
        x = f.dropout(x, self.dropout, training=self.training)
        x = f.elu(self.out_layer(x, adj))
        output = f.log_softmax(x, dim=1)
        return output
