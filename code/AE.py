import torch
import torch.nn as nn
import torch.nn.functional as F


def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(input_size, hidden_size)
        self.decoder = Decoder(hidden_size, output_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = F.mish(x)
        return x


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        encoder = []
        for i in range(len(input_size)-1):
            encoder.append(nn.Linear(input_size[i], input_size[i+1]))
        encoder.append(nn.Linear(input_size[-1], hidden_size))
        self.encoders = nn.Sequential(*encoder)
        self.encoders.apply(xavier_init)

    def forward(self, x):
        x = F.mish(self.encoders(x))
        return x


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        decoder = [nn.Linear(hidden_size, output_size[0])]
        for i in range(len(output_size)-1):
            decoder.append(nn.Linear(output_size[i], output_size[i+1]))
        self.decoders = nn.Sequential(*decoder)
        self.decoders.apply(xavier_init)

    def forward(self, x):
        x = F.mish(self.decoders(x))
        return x
