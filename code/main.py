from train_test import train_test

if __name__ == "__main__":
    # data_folder = 'E:\chenyuqi\MOGAD\ROSMAP\\no_feature_selection\\6'
    data_folder = 'F:\graduate\\run\ROSMAP\\no_feature_selection\\6'
    # data_folder = 'E:\chenyuqi\MOGAD\BRCA\clear\z_score\\1'
    # data_folder = 'F:\graduate\\run\BRCA\clear\z_score\\1'
    view_list = [1, 2, 3]
    num_epoch_pretrain = 500
    num_epoch = 2500
    config = {}
    clinical = False
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

    train_test(data_folder, view_list, num_class, config, num_epoch_pretrain, num_epoch, clinical, train_AE)

