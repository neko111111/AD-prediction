import torch
from torch import nn
from torch.nn import functional as f
from Self_Attention import Self_Attention, Self_Attention_Muti_Head
from GAT import GAT
from GCN import GCN_E
from VAE import VAE
from SNN import SpikingNeuralNetwork


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
        output = f.normalize(output)
        return output


class Classifier_1(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        x = f.normalize(x)
        return x


# 参数：组学数据的数量，预测种类，隐藏层维度
class VCDN(nn.Module):
    def __init__(self, num_view, num_cls, hvcdn_dim):
        super().__init__()
        self.num_cls = num_cls
        self.model = nn.Sequential(
            nn.Linear(pow(num_cls, num_view), hvcdn_dim),
            nn.LeakyReLU(0.25),
            nn.Linear(hvcdn_dim, num_cls)
        )
        self.model.apply(xavier_init)

    def forward(self, in_list):
        num_view = len(in_list)
        for i in range(num_view):
            in_list[i] = torch.sigmoid(in_list[i])
        x = torch.reshape(torch.matmul(in_list[0].unsqueeze(-1), in_list[1].unsqueeze(1)),
                          (-1, pow(self.num_cls, 2), 1))
        for i in range(2, num_view):
            x = torch.reshape(torch.matmul(x, in_list[i].unsqueeze(1)), (-1, pow(self.num_cls, i + 1), 1))
        vcdn_feat = torch.reshape(x, (-1, pow(self.num_cls, num_view)))
        output = self.model(vcdn_feat)

        return output


# num_view:组学数据的数量(3), num_class:预测类别数量,
# dim_list:三个组学数据的特征值数量列表([200,200,200]), dim_he_list:[200,200,100],
# dim_hc:pow(num_class,num_view)
def init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hc):
    model_dict = {}
    for i in range(num_view):
        model_dict["A{:}".format(i + 1)] = VAE([200, 150], 100, [150, 200])
        # model_dict["B{:}".format(i + 1)] = GAT(dim_list[i], 100, 50, 0.5, 0.2, 4)
        model_dict["C{:}".format(i + 1)] = Classifier_1(200, num_class)
    if num_view >= 2:
        model_dict["C"] = VCDN(num_view, num_class, dim_hc)
    return model_dict


def init_optim(num_view, model_dict, lr_e, lr_c=1e-4):
    optim_dict = {}
    for i in range(num_view):
        optim_dict["C{:}".format(i + 1)] = torch.optim.Adam(
            list(model_dict["A{:}".format(i + 1)].parameters()) +
            # list(model_dict["B{:}".format(i + 1)].parameters()) +
            list(model_dict["C{:}".format(i + 1)].parameters()),
            lr=lr_e)
    if num_view >= 2:
        optim_dict["C"] = torch.optim.Adam(model_dict["C"].parameters(), lr=lr_c)
    return optim_dict
