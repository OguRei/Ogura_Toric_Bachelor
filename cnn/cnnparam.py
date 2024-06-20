import numpy as np
import os
import sys
sys.path.append(os.path.abspath("../"))
from hyperparam import HyperParam as hp 
class CnnParam():
        
    use_fp16 = False

    datafile_path = f'/home/mukailab/test/ogura_code/MyFiles/datasets/dataset_{hp.num_data}_size{hp.code_distance}.pt'
    model_dir = "/home/mukailab/test/ogura_code/MyFiles/models"
    #dataset_dir = f"/datasets/"
    use_plane = True #プレーンデータを使うかどうか
    log_per = 1 #学習時にログを表示する頻度
    # dataset_algo = "entropy" #エラーを発生させる数を決める手法の決定 utils
    dataset_runtime = False #Trueなら生成しながら学習 tsubota toriccode

    dataset_size_train = 1.09 / 1.1

    dataset_algo = "entropy"
    distinct = False
    min_errors = 2
    max_errors = 30
    batch_size = 256 #256
    n_epoch = 100 #1000
    optuna_epoch = 100
    channels = 256

    lr = 5e-4 #学習率
    betas = (0.9, 0.999)
    momentum = 0.9
    eps = 1e-6 if use_fp16 else 1e-8

    # more greater, more trustable
    minimum_threshold = 0.3
    resnet_layers = 8

    nhead = 4
    pe_auto = False
    kernel_size = 2
    activation = "mish"
    activation_args = {}
    batch_norm = True
    dp = 0.1
    weight_init = "normal"
    padding = "zeros"

    flooding = 0.0
    noise_max = 0.0
    use_optuna =False
# CNN用