from torch import nn
from torch.nn import functional as F


def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


# 全连接层，参数(输入特征数，中间特征数(列表)，输出特征数，激活函数)
class FullConnect(nn.Module):
    def __init__(self, in_features, latent_features, out_features, activity=nn.ReLU()):
        super().__init__()
        n = len(latent_features)
        self.activity = activity
        if n != 0:
            fc = [nn.Linear(in_features, latent_features[0]), self.activity]
            for i in range(n - 1):
                fc.append(nn.Linear(latent_features[i], latent_features[i + 1]))
                fc.append(self.activity)
                fc.append(nn.Dropout(0.5))
            fc.append(nn.Linear(latent_features[-1], out_features))
        else:
            fc = [nn.Linear(in_features, out_features)]
        self.model = nn.Sequential(*fc)
        self.model.apply(xavier_init)

    def forward(self, x):
        output = self.model(x)
        output = F.normalize(output)
        return output


class Classifier_1(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        x = F.normalize(x)
        return x

