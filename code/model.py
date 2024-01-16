from torch import nn
from GAT import GAT
from AE import Encoder, Decoder


class model(nn.Module):
    def __init__(self, in_dim, hgcn_dim):
        super().__init__()
        self.e1 = Encoder([in_dim, (in_dim+hgcn_dim[0])//2], hgcn_dim[0])
        self.gat1 = GAT(in_dim, (in_dim+hgcn_dim[0])//2, hgcn_dim[0], 0.5, 2)
        self.e2 = Encoder([hgcn_dim[0], (hgcn_dim[0]+hgcn_dim[1])//2], hgcn_dim[1])
        self.gat2 = GAT(hgcn_dim[0], (hgcn_dim[0]+hgcn_dim[1])//2, hgcn_dim[1], 0.5, 2)
        self.d = Decoder(hgcn_dim[1], [hgcn_dim[0], in_dim])

    def forward(self, x, adj, e):
        z = x
        x = self.e1(x)
        z = self.gat1(z, adj)
        x = (1-e) * x + e * z

        x = self.e2(x)
        z = self.gat2(z, adj)
        x = (1 - e) * x + e * z

        x = self.d(x)

        return x
