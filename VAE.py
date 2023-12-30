import torch
import torch.nn as nn


def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


class VAE(nn.Module):
    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes):
        super().__init__()
        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list
        self.latent_size = latent_size
        self.encoder = Encoder(encoder_layer_sizes, latent_size)
        self.decoder = Decoder(decoder_layer_sizes, latent_size)

    def forward(self, x):
        means_encoder, log_var_encoder = self.encoder(x)
        z = self.reparameterize(means_encoder, log_var_encoder)    # Z = means + eps*exp(log_var/2)
        recon_x = self.decoder(z)
        return recon_x

    def reparameterize(self, means, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return means + eps * std


class Encoder(nn.Module):
    def __init__(self, layer_sizes, latent_size):
        super().__init__()
        mlp = []
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            mlp.append(nn.Linear(in_size, out_size))
            mlp.append(nn.BatchNorm1d(out_size))
            mlp.append(nn.ReLU())
        self.MLP = nn.Sequential(*mlp)
        self.MLP.apply(xavier_init)
        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x):
        x = self.MLP(x)
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        return means, log_vars


class Decoder(nn.Module):
    def __init__(self, layer_sizes, latent_size):
        super().__init__()
        input_size = latent_size
        mlp = []
        for i, (in_size, out_size) in enumerate(zip([input_size] + layer_sizes[:-1], layer_sizes)):
            mlp.append(nn.Linear(in_size, out_size))
            mlp.append(nn.BatchNorm1d(out_size))
            if i + 1 < len(layer_sizes):
                mlp.append(nn.ReLU())
            else:
                mlp.append(nn.Tanh())
        self.MLP = nn.Sequential(*mlp)
        self.MLP.apply(xavier_init)

    def forward(self, z):
        x = self.MLP(z)
        return x
