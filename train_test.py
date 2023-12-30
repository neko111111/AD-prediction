""" Training and testing of the model
"""
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
import torch.nn.functional as F
from model import init_model_dict, init_optim
from utils import one_hot_tensor, cal_sample_weight, cal_adj_mat_parameter, gen_adj_mat_tensor, gen_test_adj_mat_tensor

cuda = True if torch.cuda.is_available() else False
if cuda:
    print('cuda')
else:
    print('cpu')


# 输入数据文件名(ROSMAP_2，BRCA).输出训练集数据(三个Tensor的集合,每个Tensor为245*200)
# 所有数据(三个Tensor的集合,每个Tensor为351*200)，两个集的序号(tr:{}, te:{})，标签列表(所有)
def prepare_trte_data(data_folder, view_list):
    num_view = len(view_list)
    labels_tr = np.loadtxt(os.path.join(data_folder, "labels_tr.csv"), delimiter=',')
    labels_te = np.loadtxt(os.path.join(data_folder, "labels_te.csv"), delimiter=',')
    labels_tr = labels_tr.astype(int)
    labels_te = labels_te.astype(int)
    data_tr_list = []
    data_te_list = []
    for i in view_list:
        data_tr_list.append(np.loadtxt(os.path.join(data_folder, str(i)+"_tr.csv"), delimiter=','))
        data_te_list.append(np.loadtxt(os.path.join(data_folder, str(i)+"_te.csv"), delimiter=','))
    num_tr = data_tr_list[0].shape[0]
    num_te = data_te_list[0].shape[0]
    data_mat_list = []
    for i in range(num_view):
        data_mat_list.append(np.concatenate((data_tr_list[i], data_te_list[i]), axis=0))
    data_tensor_list = []
    for i in range(len(data_mat_list)):
        data_tensor_list.append(torch.FloatTensor(data_mat_list[i]))
        if cuda:
            data_tensor_list[i] = data_tensor_list[i].cuda()
    idx_dict = {}
    idx_dict["tr"] = list(range(num_tr))
    idx_dict["te"] = list(range(num_tr, (num_tr+num_te)))
    data_train_list = []
    data_all_list = []
    for i in range(len(data_tensor_list)):
        data_train_list.append(data_tensor_list[i][idx_dict["tr"]].clone())
        data_all_list.append(torch.cat((data_tensor_list[i][idx_dict["tr"]].clone(),
                                       data_tensor_list[i][idx_dict["te"]].clone()),0))
    labels = np.concatenate((labels_tr, labels_te))
    
    return data_train_list, data_all_list, idx_dict, labels


# 求邻接矩阵列表
def gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter):
    adj_metric = "cosine"  # cosine distance
    adj_train_list = []
    adj_test_list = []
    for i in range(len(data_tr_list)):
        adj_parameter_adaptive = cal_adj_mat_parameter(adj_parameter, data_tr_list[i], adj_metric)
        adj_train_list.append(gen_adj_mat_tensor(data_tr_list[i], adj_parameter_adaptive, adj_metric))
        adj_test_list.append(gen_test_adj_mat_tensor(data_trte_list[i], trte_idx, adj_parameter_adaptive, adj_metric))

    return adj_train_list, adj_test_list


# 输入：训练集数据，训练集标签，标签的one hot形式，标签占比列表，模型字典，优化器字典
def train_epoch(data_list, label, one_hot_label, sample_weight, adj_list, model_dict, optim_dict, train_VCDN=True):
    loss_dict = {}
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    for m in model_dict:
        model_dict[m].train()   # 训练模型
    num_view = len(data_list)   # 组学数据个数
    for i in range(num_view):
        optim_dict["C{:}".format(i+1)].zero_grad()
        ci_loss = 0
        ci = model_dict["C{:}".format(i+1)](model_dict["A{:}".format(i+1)](data_list[i]))  # x = Ci(Ei(Di(x))
        ci_loss = torch.mean(torch.mul(criterion(ci, label), sample_weight))
        ci_loss.backward()
        optim_dict["C{:}".format(i+1)].step()
        loss_dict["C{:}".format(i+1)] = ci_loss.detach().cpu().numpy().item()
    c = 0
    if train_VCDN and num_view >= 2:
        optim_dict["C"].zero_grad()
        c_loss = 0
        ci_list = []
        for i in range(num_view):
            ci_list.append(model_dict["C{:}".format(i+1)](model_dict["A{:}".format(i+1)](data_list[i])))
        c = model_dict["C"](ci_list)
        c_loss = torch.mean(torch.mul(criterion(c, label), sample_weight))
        c_loss.backward()
        optim_dict["C"].step()
        loss_dict["C"] = c_loss.detach().cpu().numpy().item()
    
    return loss_dict
    

def test_epoch(data_list, te_idx, adj_list, model_dict):
    for m in model_dict:
        model_dict[m].eval()
    num_view = len(data_list)
    ci_list = []
    for i in range(num_view):
        ci_list.append(model_dict["C{:}".format(i+1)](model_dict["A{:}".format(i+1)](data_list[i])))
    if num_view >= 2:
        c = model_dict["C"](ci_list)
    else:
        c = ci_list[0]
    c = c[te_idx, :]
    prob = F.softmax(c, dim=1).data.cpu().numpy()
    
    return prob


def train_test(data_folder, view_list, num_class,
               lr_e_pretrain, lr_e, lr_c, 
               num_epoch_pretrain, num_epoch):
    test_inverval = 250
    adj_parameter = 2
    dim_he_list = [100, 100, 100]
    num_view = len(view_list)
    dim_hvcdn = pow(num_class, num_view)
    # 训练集数据，所有数据，两个集的序号，标签
    data_tr_list, data_trte_list, trte_idx, labels_trte = prepare_trte_data(data_folder, view_list)
    # labels_trte[trte_idx["tr"]]:训练集标签列表
    # labels_trte[trte_idx["te"]]:测试集标签列表
    # onehot_labels_tr_tensor: 离散类别的one_hot编码
    # sample_weight_tr: 标签占比
    labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])
    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class)
    sample_weight_tr = cal_sample_weight(labels_trte[trte_idx["tr"]], num_class)
    sample_weight_tr = torch.FloatTensor(sample_weight_tr)
    if cuda:
        labels_tr_tensor = labels_tr_tensor.cuda()
        onehot_labels_tr_tensor = onehot_labels_tr_tensor.cuda()
        sample_weight_tr = sample_weight_tr.cuda()
    adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter)  # 求邻接矩阵
    dim_list = [x.shape[1] for x in data_tr_list]  # 训练集列数列表
    model_dict = init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hvcdn)
    for m in model_dict:
        if cuda:
            model_dict[m].cuda()
    
    print("\nPretrain Networks...")
    optim_dict = init_optim(num_view, model_dict, lr_e_pretrain, lr_c)
    for epoch in range(num_epoch_pretrain):
        train_epoch(data_tr_list, labels_tr_tensor,
                    onehot_labels_tr_tensor, sample_weight_tr, adj_tr_list, model_dict, optim_dict, train_VCDN=False)
    print("\nTraining...")
    optim_dict = init_optim(num_view, model_dict, lr_e, lr_c)
    for epoch in range(num_epoch+1):
        loss = train_epoch(data_tr_list, labels_tr_tensor,
                    onehot_labels_tr_tensor, sample_weight_tr, adj_tr_list, model_dict, optim_dict)
        if epoch % test_inverval == 0:
            print(loss)
            te_prob = test_epoch(data_trte_list, trte_idx["te"], adj_te_list, model_dict)
            # tr_prob = test_epoch(data_trte_list, trte_idx["tr"], model_dict)
            print("\nTest: Epoch {:d}".format(epoch))
            if num_class == 2:
                # print("Train ACC: {:.3f}".format(accuracy_score(labels_trte[trte_idx["tr"]], tr_prob.argmax(1))))
                # print("Train F1: {:.3f}".format(f1_score(labels_trte[trte_idx["tr"]], tr_prob.argmax(1))))
                # print("Train AUC: {:.3f}".format(roc_auc_score(labels_trte[trte_idx["tr"]], tr_prob[:, 1])))
                print("Test ACC: {:.3f}".format(accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
                print("Test F1: {:.3f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
                print("Test AUC: {:.3f}".format(roc_auc_score(labels_trte[trte_idx["te"]], te_prob[:,1])))
            else:
                print("Test ACC: {:.3f}".format(accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
                print("Test F1 weighted: {:.3f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='weighted')))
                print("Test F1 macro: {:.3f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')))
            print()

# print(init_model_dict(3, 2, [200,200,200], 100, 50, 50, 8))
