import os
import copy
from feat_importance import cal_feat_imp, summarize_imp_feat

if __name__ == "__main__":
    # data_folder = 'F:\graduate\\run\BRCA\clear\z_score\\1'
    data_folder = 'F:\graduate\\run\ROSMAP\\no_feature_selection\\1'
    model_folder = 'F:\graduate\models\ROSMAP\\1\\ALLRegion'
    view_list = [1, 'ALLRegion', 3]
    folder = 'ALLRegion'
    num_biomarker = 300
    config = {}

    if "ROSMAP" in data_folder:
        train_AE = True
        config['hid_AE'] = 100
        config['out_AE'] = [200]
        config['dim_k'] = 200
        config['dim_v'] = config['out_AE'][-1]
        config['in_VCDN'] = config['out_AE'][-1]
        config['hid_VCDN'] = 50
        config['dropout_AE'] = 0
        config['dropout_SA'] = 0.5
        config['dropout_VCDN'] = 0.8
        config['lr_a_pretrain'] = 1e-3
        config['lr_b_pretrain'] = 1e-4
        config['lr_a'] = 5e-4
        config['lr_b'] = 5e-5
        config['lr_c'] = 5e-4
        num_class = 2
    if "BRCA" in data_folder:
        train_AE = False
        config['hid_AE'] = 100
        config['out_AE'] = [200]
        config['dim_k'] = 200
        config['dim_v'] = config['out_AE'][-1]
        config['in_VCDN'] = config['out_AE'][-1]
        config['hid_VCDN'] = 50
        config['dropout_AE'] = 0
        config['dropout_SA'] = 0.1
        config['dropout_VCDN'] = 0.8
        config['lr_a_pretrain'] = 1e-5
        config['lr_b_pretrain'] = 1e-5
        config['lr_a'] = 5e-5
        config['lr_b'] = 5e-6
        config['lr_c'] = 5e-4
        num_class = 5

    featimp_list_list = []
    for rep in range(5):
        featimp_list = cal_feat_imp(data_folder, os.path.join(model_folder, str(rep + 1)),
                                    view_list, num_class, rep + 1, config, train_AE)
        featimp_list_list.append(copy.deepcopy(featimp_list))
    summarize_imp_feat(featimp_list_list, folder, num_biomarker)
