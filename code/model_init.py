import torch
from model import model
from FullConnect import Classifier_1
from VCDN import VCDN


# num_view:组学数据的数量(3), num_class:预测类别数量,
# dim_list:三个组学数据的特征值数量列表([200,200,200]), dim_he_list:[200,200,100],
# dim_hc:pow(num_class,num_view)
def init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hc):
    model_dict = {}
    for i in range(num_view):
        # model_dict["A{:}".format(i + 1)] = VAE([200, 150], 100, [150, 200])
        # model_dict["B{:}".format(i + 1)] = GAT(dim_list[i], 100, 2, 0.5, 0.2, 4)
        model_dict["B{:}".format(i + 1)] = model(dim_list[i], dim_he_list)
        model_dict["C{:}".format(i + 1)] = Classifier_1(dim_list[i], num_class)
    if num_view >= 2:
        model_dict["C"] = VCDN(num_view, num_class, dim_hc)
    return model_dict


def init_optim(num_view, model_dict, lr_e, lr_c=1e-4):
    optim_dict = {}
    for i in range(num_view):
        optim_dict["C{:}".format(i + 1)] = torch.optim.Adam(
            # list(model_dict["A{:}".format(i + 1)].parameters()) +
            list(model_dict["B{:}".format(i + 1)].parameters()) +
            list(model_dict["C{:}".format(i + 1)].parameters()),
            lr=lr_e)
    if num_view >= 2:
        optim_dict["C"] = torch.optim.Adam(model_dict["C"].parameters(), lr=lr_c)
    return optim_dict
