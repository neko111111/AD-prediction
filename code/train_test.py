""" Training and testing of the model
"""
import copy
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, matthews_corrcoef
import torch
import torch.nn.functional as F
from model_init import init_model_dict, init_optim
from utils import one_hot_tensor, cal_sample_weight, cal_adj_mat_parameter, gen_adj_mat_tensor, gen_test_adj_mat_tensor, save_model_dict
# from HyperGragh import construct_H_with_KNN, generate_G_from_H

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
print(device)


# 输入数据文件名(ROSMAP_2，BRCA).输出训练集数据(三个Tensor的集合,每个Tensor为245*200)
# 所有数据(三个Tensor的集合,每个Tensor为351*200)，两个集的序号(tr:{}, te:{})，标签列表(所有)
def prepare_trte_data(data_folder, view_list, clinical=False):
    num_view = len(view_list)
    labels_tr = np.loadtxt(os.path.join(data_folder, f"labels_tr.csv"), delimiter=',')
    labels_te = np.loadtxt(os.path.join(data_folder, f"labels_te.csv"), delimiter=',')
    labels_tr = labels_tr.astype(int)
    labels_te = labels_te.astype(int)
    data_tr_list = []
    data_te_list = []
    num_train = len(labels_tr)

    if clinical:
        df = pd.concat([pd.read_csv(f'{data_folder}/unomics_tr.csv'), pd.read_csv(f'{data_folder}/unomics_te.csv')],
                       axis=0)
        df['apoe_genotype'] = df['apoe_genotype'].apply(
            lambda x: 0 if x == 22 else (
                0 if x == 23 else (1 if x == 24 else (0 if x == 33 else (1 if x == 34 else 1)))))
        df['ceradsc'] = df['ceradsc'] - 1
        # X = df[['apoe_genotype', 'ceradsc', 'braaksc']]
        # this
        X = df['apoe_genotype'].values.reshape(-1, 1)
        mean = X.mean()
        std = X.std()
        clinical_data = (X - mean) / std
        # scaler = MinMaxScaler()
        # clinical_data = scaler.fit_transform(X)
        clinical_data = clinical_data * 6

    for i in view_list:
        if clinical:
            data_tr_list.append(
                np.hstack(
                    (np.loadtxt(os.path.join(data_folder, str(i) + f"_tr.csv"), delimiter=','),
                     clinical_data[:num_train])))
            data_te_list.append(
                np.hstack(
                    (np.loadtxt(os.path.join(data_folder, str(i) + f"_te.csv"), delimiter=','),
                     clinical_data[num_train:])))
        else:
            data_tr_list.append(np.loadtxt(os.path.join(data_folder, str(i) + f"_tr.csv"), delimiter=','))
            data_te_list.append(np.loadtxt(os.path.join(data_folder, str(i) + f"_te.csv"), delimiter=','))
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
    idx_dict["te"] = list(range(num_tr, (num_tr + num_te)))
    data_train_list = []
    data_all_list = []
    for i in range(len(data_tensor_list)):
        data_train_list.append(data_tensor_list[i][idx_dict["tr"]].clone())
        data_all_list.append(torch.cat((data_tensor_list[i][idx_dict["tr"]].clone(),
                                        data_tensor_list[i][idx_dict["te"]].clone()), 0))
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
def train_epoch(data_list, label, one_hot_label, sample_weight, adj_list, model_dict, optim_dict, train_AE=True, train_VCDN=True):
    loss_dict = {}
    criterion_a = torch.nn.CrossEntropyLoss(reduction='none')
    criterion_b = torch.nn.CrossEntropyLoss(reduction='none')
    criterion_c = torch.nn.MSELoss(reduction='none')
    for m in model_dict:
        model_dict[m].train()  # 训练模型
    num_view = len(data_list)  # 组学数据个数

    if train_AE:
        optim_dict["A"].zero_grad()
        c_loss = 0
        c = model_dict["A"](data_list)
        c_loss = torch.mean(torch.mul(criterion_a(c, label), sample_weight))
        c_loss.backward()
        optim_dict["A"].step()
        loss_dict["A"] = c_loss.detach().cpu().numpy().item()

    for i in range(num_view):
        optim_dict["B{:}".format(i + 1)].zero_grad()
        ci_loss = 0
        ci = model_dict["B{:}".format(i + 1)](data_list[i], adj_list[i])
        ci_loss = torch.mean(torch.mul(criterion_b(ci, label), sample_weight))
        ci_loss.backward()
        optim_dict["B{:}".format(i + 1)].step()
        loss_dict["B{:}".format(i + 1)] = ci_loss.detach().cpu().numpy().item()

    c = 0
    if train_VCDN and num_view >= 2:
        optim_dict["C"].zero_grad()
        c_loss = 0
        ci_list = []
        for i in range(num_view):
            ci_list.append(
                model_dict["B{:}".format(i + 1)](data_list[i], adj_list[i]))
        if train_AE:
            ci_list.append(model_dict["A"](data_list))
        c = model_dict["C"](ci_list)
        c_loss = torch.mean(torch.mul(criterion_c(c, one_hot_label).T, sample_weight))
        c_loss.backward()
        optim_dict["C"].step()
        loss_dict["C"] = c_loss.detach().cpu().numpy().item()

    return loss_dict


def test_epoch(data_list, te_idx, adj_list, model_dict, test_AE=True):
    for m in model_dict:
        model_dict[m].eval()
    num_view = len(data_list)
    ci_list = []
    # data = torch.cat(data_list, dim=0)
    if test_AE:
        ci_list.append(model_dict["A"](data_list))
    for i in range(num_view):
        ci_list.append(
            model_dict["B{:}".format(i + 1)](data_list[i], adj_list[i]))
    if num_view >= 2:
        c = model_dict["C"](ci_list)
    else:
        c = ci_list[0]
    c = c[te_idx, :]
    prob = F.softmax(c, dim=1).data.cpu().numpy()

    return prob


def train_test(data_folder, view_list, num_class, config,
               num_epoch_pretrain, num_epoch, clinical, train_AE):
    test_inverval = 25
    adj_parameter = 2
    num_view = len(view_list)
    # 训练集数据，所有数据，两个集的序号，标签
    data_tr_list, data_trte_list, trte_idx, labels_trte = prepare_trte_data(data_folder, view_list, clinical)
    labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])
    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class)
    sample_weight_tr = cal_sample_weight(labels_trte[trte_idx["tr"]], num_class)
    sample_weight_tr = torch.FloatTensor(sample_weight_tr)
    labels_tr_tensor = labels_tr_tensor.to(device)
    onehot_labels_tr_tensor = onehot_labels_tr_tensor.to(device)
    sample_weight_tr = sample_weight_tr.to(device)
    adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter)  # 求邻接矩阵

    for i in range(len(adj_tr_list)):
        adj_tr_list[i] = adj_tr_list[i].to_dense()
        adj_te_list[i] = adj_te_list[i].to_dense()

    dim_list = [x.shape[1] for x in data_tr_list]  # 训练集列数列表

    model_dict = init_model_dict(num_view, num_class, dim_list, config['hid_AE'], config['out_AE'], config['dim_k'],
                                 config['dim_v'], config['in_VCDN'], config['hid_VCDN'], train_AE, config['dropout_AE'],
                                 config['dropout_SA'], config['dropout_VCDN'])
    for m in model_dict:
        model_dict[m].to(device)

    max_acc = 0
    max_auc = 0
    max_f1 = 0
    max_mcc = 0
    max_f1_weight = 0
    max_f1_macro = 0
    max_all = 0
    model_best = copy.deepcopy(model_dict)

    print("\nPretrain Networks...")
    optim_dict = init_optim(num_view, model_dict, config['lr_a_pretrain'], config['lr_b_pretrain'], config['lr_c'])
    for epoch in range(num_epoch_pretrain):
        train_epoch(data_tr_list, labels_tr_tensor, onehot_labels_tr_tensor, sample_weight_tr,
                    adj_tr_list, model_dict, optim_dict, train_AE=train_AE, train_VCDN=False)
    print("\nTraining...")
    optim_dict = init_optim(num_view, model_dict,  config['lr_a'],  config['lr_b'],  config['lr_c'])
    for epoch in range(num_epoch + 1):
        loss = train_epoch(data_tr_list, labels_tr_tensor, onehot_labels_tr_tensor, sample_weight_tr,
                           adj_tr_list, model_dict, optim_dict, train_AE=train_AE)
        if epoch % test_inverval == 0:
            print(loss)
            te_prob = test_epoch(data_trte_list, trte_idx["te"], adj_te_list, model_dict, train_AE)
            print(te_prob.argmax(1))
            print("\nTest: Epoch {:d}".format(epoch))
            if num_class == 2:
                acc = accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
                auc = roc_auc_score(labels_trte[trte_idx["te"]], te_prob[:, 1])
                f1 = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
                mcc = matthews_corrcoef(labels_trte[trte_idx["te"]], te_prob.argmax(1))
                print("Test ACC: {:.3f}".format(acc))
                print("Test F1: {:.3f}".format(f1))
                print("Test AUC: {:.3f}".format(auc))
                print("Test MCC: {:.3f}".format(mcc))
                if (acc + auc + f1 + mcc) >= max_all:
                    max_all = acc + auc + f1 + mcc
                    max_acc = acc
                    max_auc = auc
                    max_f1 = f1
                    max_mcc = mcc
                    model_best = copy.deepcopy(model_dict)
            else:
                acc = accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
                f1_weight = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='weighted')
                f1_macro = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')
                print("Test ACC: {:.3f}".format(acc))
                print("Test F1 weighted: {:.3f}".format(f1_weight))
                print("Test F1 macro: {:.3f}".format(f1_macro))
                if (acc + f1_macro + f1_weight) >= max_all:
                    max_all = acc + f1_macro + f1_weight
                    max_acc = acc
                    max_f1_macro = f1_macro
                    max_f1_weight = f1_weight
                    model_best = copy.deepcopy(model_dict)
            print()
    save_model_dict(f'{data_folder}\models\\5', model_best)
    if num_class == 2:
        print("MAX ACC: {:.3f}".format(max_acc))
        print("MAX F1: {:.3f}".format(max_f1))
        print("MAX AUC: {:.3f}".format(max_auc))
        print("MAX MCC: {:.3f}".format(max_mcc))
        print("MAX ALL: {:.3f}".format(max_all))
    else:
        print("MAX ACC: {:.3f}".format(max_acc))
        print("MAX F1 weighted: {:.3f}".format(max_f1_weight))
        print("MAX F1 macro: {:.3f}".format(max_f1_macro))
        print("MAX ALL: {:.3f}".format(max_all))

