import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.sparse import diags
import pandas as pd
from pathlib import Path
import numpy as np
import sklearn
import os
import torch
import torch.nn.functional as F

cuda = True if torch.cuda.is_available() else False


# 除去dataframe中平均值小于等于mean_s和方差小于等于var_s的列
def filter_mean_var(data, mean_s, var_s):
    # 按列算平均值和方差
    ave = pd.DataFrame(data.mean(numeric_only=True)).reset_index().rename(columns={'index': 'feature', 0: 'value'})
    var = pd.DataFrame(data.var(numeric_only=True)).reset_index().rename(columns={'index': 'feature', 0: 'value'})
    ave = ave.loc[ave.value <= mean_s]
    var = var.loc[var.value <= var_s]
    arr = ave.feature.tolist()
    arr.extend(var.feature.tolist())
    arr = list(set(arr))
    data.drop(arr, inplace=True, axis=1)
    return data


# anova-F值(数据，自变量，因变量)
def anova(data, independent_variable, dependent_variable):
    lm = ols(dependent_variable + ' ~ ' + independent_variable, data=data).fit()
    table = sm.stats.anova_lm(lm, typ=1)
    return table


# DataFrame归一化
def normalize(mx):
    scaler = sklearn.preprocessing.MinMaxScaler()
    scaler.fit(mx)
    mx = scaler.transform(mx)
    mx = pd.DataFrame(mx)
    return mx


# 加载数据或标签并转换成tensor，参数为文件名
# def load_data(name):
#     folder_name = path / name
#     mx = np.loadtxt(os.path.join(folder_name), delimiter=',')
#     mx = np.array(mx)
#     mx = torch.FloatTensor(mx)
#     return mx


def one_hot_tensor(y, num_dim):
    y_onehot = torch.zeros(y.shape[0], num_dim)
    y_onehot.scatter_(1, y.view(-1, 1), 1)

    return y_onehot


# sample_weight表示标签占比
def cal_sample_weight(labels, num_class, use_sample_weight=True):
    if not use_sample_weight:
        return np.ones(len(labels)) / len(labels)
    count = np.zeros(num_class)
    for i in range(num_class):
        count[i] = np.sum(labels == i)  # count[0]为标签为0的样本个数，count[1]为标签为1的样本个数。
    sample_weight = np.zeros(labels.shape)
    for i in range(num_class):
        sample_weight[np.where(labels == i)[0]] = count[i] / np.sum(count)

    return sample_weight


def cosine_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def to_sparse(x):
    x_typename = torch.typename(x).split('.')[-1]
    print(x_typename)
    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return torch.sparse_coo_tensor(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return torch.sparse_coo_tensor(indices, values, x.size())


def cal_adj_mat_parameter(edge_per_node, data, metric="cosine"):
    assert metric == "cosine", "Only cosine distance implemented"
    dist = cosine_distance_torch(data, data)
    parameter = torch.sort(dist.reshape(-1, )).values[edge_per_node * data.shape[0]]
    return np.ndarray.item(parameter.data.cpu().numpy())


def graph_from_dist_tensor(dist, parameter, self_dist=True):
    if self_dist:
        assert dist.shape[0] == dist.shape[1], "Input is not pairwise dist matrix"
    g = (dist <= parameter).float()
    if self_dist:
        diag_idx = np.diag_indices(g.shape[0])
        g[diag_idx[0], diag_idx[1]] = 0

    return g


def gen_adj_mat_tensor(data, parameter, metric="cosine"):
    assert metric == "cosine", "Only cosine distance implemented"
    dist = cosine_distance_torch(data, data)
    g = graph_from_dist_tensor(dist, parameter, self_dist=True)
    if metric == "cosine":
        adj = 1 - dist
    else:
        raise NotImplementedError
    adj = adj * g
    adj_T = adj.transpose(0, 1)
    I = torch.eye(adj.shape[0])
    if cuda:
        I = I.cuda()
    adj = adj + adj_T * (adj_T > adj).float() - adj * (adj_T > adj).float()
    adj = F.normalize(adj + I, p=1)
    # adj = to_sparse(adj)

    return adj


def gen_test_adj_mat_tensor(data, trte_idx, parameter, metric="cosine"):
    assert metric == "cosine", "Only cosine distance implemented"
    adj = torch.zeros((data.shape[0], data.shape[0]))
    if cuda:
        adj = adj.cuda()
    num_tr = len(trte_idx["tr"])

    dist_tr2te = cosine_distance_torch(data[trte_idx["tr"]], data[trte_idx["te"]])
    g_tr2te = graph_from_dist_tensor(dist_tr2te, parameter, self_dist=False)
    if metric == "cosine":
        adj[:num_tr, num_tr:] = 1 - dist_tr2te
    else:
        raise NotImplementedError
    adj[:num_tr, num_tr:] = adj[:num_tr, num_tr:] * g_tr2te

    dist_te2tr = cosine_distance_torch(data[trte_idx["te"]], data[trte_idx["tr"]])
    g_te2tr = graph_from_dist_tensor(dist_te2tr, parameter, self_dist=False)
    if metric == "cosine":
        adj[num_tr:, :num_tr] = 1 - dist_te2tr
    else:
        raise NotImplementedError
    adj[num_tr:, :num_tr] = adj[num_tr:, :num_tr] * g_te2tr  # retain selected edges

    adj_T = adj.transpose(0, 1)
    I = torch.eye(adj.shape[0])
    if cuda:
        I = I.cuda()
    adj = adj + adj_T * (adj_T > adj).float() - adj * (adj_T > adj).float()
    adj = F.normalize(adj + I, p=1)
    # adj = to_sparse(adj)

    return adj
