import torch
from AE import AutoEncoder
from Self_Attention import MultiHeadSelfAttention, SA_Fusion, CrossAttention
from transformer import SimpleTransformerClassifier
from FullConnect import Classifier_1
from encoder import Encoder
from VCDN import VCDN
from attention_fusion import AttentionFusion, NNFusion
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_model_dict(num_view, num_class, dim_list, hid_AE, out_AE, dim_k, dim_v,
                    in_VCDN, hid_VCDN, train_AE, dropout_AE, dropout_SA, dropout_VCDN):
    model_dict = {}
    model_dict["A"] = AutoEncoder(num_view, dim_list, hid_AE, out_AE, dropout_AE)
    for i in range(num_view):
        model_dict["B{:}".format(i + 1)] = CrossAttention(dim_list[i], dim_k, dim_v, 10, dropout_SA)
    if num_view >= 2:
        if train_AE:
            model_dict["C"] = AttentionFusion(in_VCDN, hid_VCDN, num_view+1, num_class, dropout_VCDN)
        else:
            model_dict["C"] = AttentionFusion(in_VCDN, hid_VCDN, num_view, num_class, dropout_VCDN)
        # model_dict["C"] = NNFusion(in_VCDN, hid_VCDN, num_view, num_class, dropout_VCDN)
    return model_dict


def init_optim(num_view, model_dict, lr_a, lr_b, lr_c):
    optim_dict = {}
    optim_dict["A"] = torch.optim.Adam(model_dict["A"].parameters(), lr=lr_a)
    for i in range(num_view):
        optim_dict["B{:}".format(i + 1)] = torch.optim.Adam(model_dict["B{:}".format(i + 1)].parameters(), lr=lr_b)
    if num_view >= 2:
        optim_dict["C"] = torch.optim.Adam(model_dict["C"].parameters(), lr=lr_c)
    return optim_dict
