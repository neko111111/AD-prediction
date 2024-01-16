from train_test import train_test

if __name__ == "__main__":
    data_folder = 'ROSMAP_2000'
    view_list = [1, 2, 3]
    num_epoch_pretrain = 500
    num_epoch = 2500
    lr_e_pretrain = 0.01
    lr_e = 0.003
    lr_c = 0.005
    num_class = 2

    train_test(data_folder, view_list, num_class,
               lr_e_pretrain, lr_e, lr_c,
               num_epoch_pretrain, num_epoch)

