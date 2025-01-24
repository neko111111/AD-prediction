import torch
import torch.nn as nn
import torch.nn.functional as F


def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


class AutoEncoder(nn.Module):
    def __init__(self, num, input_size, hidden_size, output_size, dropout_rate):
        super(AutoEncoder, self).__init__()
        self.a = nn.Parameter(torch.rand(num))
        # self.encoder = Encoder([input_size], hidden_size)
        encoder = []
        for i in range(num):
            encoder.append(Encoder([input_size[i]], hidden_size))
        self.encoders = nn.Sequential(*encoder)
        self.decoder = Decoder(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x_list):
        # res = []
        # for i, x in enumerate(x_list):
        #     res.append(self.a[i] * x)
        # res = torch.cat([b for b in res], dim=1)
        # res = self.encoder(res)
        # res = self.decoder(res)
        # res = self.dropout(res)
        # return F.softmax(res, dim=1)
        res = self.a[0] * self.encoders[0](x_list[0])
        for i in range(1, len(x_list)):
            res += self.a[i] * self.encoders[i](x_list[i])

        res = self.decoder(res)
        res = self.dropout(res)
        return F.softmax(res, dim=1)


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        encoder = []
        for i in range(len(input_size)-1):
            encoder.append(nn.Linear(input_size[i], input_size[i+1]))
        encoder.append(nn.Linear(input_size[-1], hidden_size))
        self.encoders = nn.Sequential(*encoder)

    def forward(self, x):
        x = F.mish(self.encoders(x))
        return x
#
# class Encoder(nn.Module):
#     def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout):
#         super(Encoder, self).__init__()
#
#         self.embedding = nn.Linear(input_dim, emb_dim)
#         self.rnn = nn.GRU(emb_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, src):
#         embedded = self.dropout(self.embedding(src))
#         embedded = embedded.unsqueeze(0)
#         outputs, hidden = self.rnn(embedded)
#         outputs = outputs.squeeze(0)
#         return outputs  # 返回最后的隐藏状态作为特征


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        decoder = [nn.Linear(hidden_size, output_size[0])]
        for i in range(len(output_size)-1):
            decoder.append(nn.Linear(output_size[i], output_size[i+1]))
        self.decoders = nn.Sequential(*decoder)

    def forward(self, x):
        x = F.mish(self.decoders(x))
        return x
