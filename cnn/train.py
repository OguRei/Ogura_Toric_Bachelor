import math
import argparse
from datetime import datetime
from multiprocessing import Pool 
import os
import random
import sys
from typing import List, Optional

import optuna
import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
import torch_optimizer
from tqdm import tqdm, trange
from cnn import CNN
from cnnparam import CnnParam as cp

sys.path.append(os.path.abspath("../"))
#from cnn import CNN
from hyperparam import HyperParam as hp
from toric_param import toric_param
from toric_code_ogura import ToricCode as TCO
from toric_code_torch import ToricCode as TCT
from makedataset import ToricCodeDataset
from utils import load_model
#from utils import ToricCodeWithTorchDataset, ToricCodeDatasetRuntime, load_model, preprocess

#cnnフォルダ入ってから実行想定

def train(cnn: CNN, #X用とZ用で分ける
          train_loader: data.DataLoader,
          toric_code: TCT,
          optimizer: optim.Optimizer,
          device = "cuda") -> List[float]: 
    errors_criterion = nn.BCEWithLogitsLoss() #二項交差エントロピーloss
    syndrome_criterion = nn.L1Loss()
    cnn.train()
    loss_list = [0.0, 0.0, 0.0, 0.0] #errorとsyndromeXだけ出してたから２つあった

    for errors, errors_X, errors_Z, syndX, syndZ in tqdm(train_loader):
        #print(errors.size())
        #print(syndX.size())
        #print(syndZ.size())

        errors = errors.to(device)

        #syndX to errorZ train
        optimizer.zero_grad(set_to_none=True)
        syndX = syndX.to(device)
        syndZ = syndZ.to(device)
        errors_X = errors_X.to(device)
        errors_Z = errors_Z.to(device)
        predicted_Z = cnn(syndX)

        #sig_predicted_Z = torch.sigmoid(predicted_Z)
        #sig_predicted_Z = torch.cat([sig_predicted_Z[:, 0:1, :, :],sig_predicted_Z[:,1:2,:,:]], 2)
        #predicted_syndrome_X = toric_code.generate_syndrome_X_differentiable(sig_predicted_Z).to(device)
        #print(predicted_syndrome_X[:2])
        predicted_Z = torch.flatten(predicted_Z)
        errors_Z = torch.flatten(errors_Z)
        loss_errors_Z = errors_criterion(predicted_Z, errors_Z)
        #predicted_syndrome_X = torch.flatten(predicted_syndrome_X)
        syndX = torch.flatten(syndX)
        #with torch.no_grad():
            #loss_syndrome_X = syndrome_criterion(predicted_syndrome_X, syndX)
        loss_list[0] += loss_errors_Z.item()
        #loss_list[1] += loss_syndrome_X.item()

        loss = loss_errors_Z
        loss.backward()
        optimizer.step() 

        #syndz to errorX train
        predicted_X = cnn(syndZ)
        sig_predicted_X = torch.sigmoid(predicted_X)
        sig_predicted_X = torch.cat([sig_predicted_X[:, 0:1, :, :],sig_predicted_X[:,1:2,:,:]], 2)
        #predicted_syndrome_Z = toric_code.generate_syndrome_Z_differentiable(sig_predicted_X).to(device)
        predicted_X = torch.flatten(predicted_X)
        errors_X = torch.flatten(errors_X)
        loss_errors_X = errors_criterion(predicted_X, errors_X)
        #predicted_syndrome_Z = torch.flatten(predicted_syndrome_Z)
        syndZ = torch.flatten(syndZ)
        #with torch.no_grad():
            #loss_syndrome_Z = syndrome_criterion(predicted_syndrome_Z, syndZ)
        loss_list[2] += loss_errors_X.item()
        #loss_list[3] += loss_syndrome_Z.item()

        loss = loss_errors_X
        loss.backward()
        optimizer.step()

    if device == "cuda":
        torch.cuda.empty_cache()
    del loss
        
    loss_list[0] /= len(train_loader)
    loss_list[1] /= len(train_loader)
    loss_list[2] /= len(train_loader)
    loss_list[3] /= len(train_loader)
    return loss_list
#[0:errorZ, 1:syndX, 2:errorX, 3:syndZ]

def evaluate(
        cnn: CNN,
        val_loader: data.DataLoader,
        toric_code: TCT,
        device = "cuda") -> List[float]:
    errors_criterion = nn.BCEWithLogitsLoss()
    syndrome_criterion = nn.L1Loss() 
    cnn.eval()
    loss_list = [0.0, 0.0, 0.0, 0.0]
    for errors, errors_X, errors_Z,syndX, syndZ in tqdm(val_loader):
        syndX = syndX.to(device)
        syndZ = syndZ.to(device)
        errors = errors.to(device)
        errors_X = errors_X.to(device)
        errors_Z = errors_Z.to(device)
        predicted_X = cnn(syndZ)
        predicted_Z = cnn(syndX)
        #sig_predicted_X = torch.sigmoid(predicted_X)
        #sig_predicted_X = torch.cat([sig_predicted_X[:, 0:1, :, :],sig_predicted_X[:,1:2,:,:]], 2)
        #sig_predicted_Z = torch.sigmoid(predicted_Z)
        #sig_predicted_Z = torch.cat([sig_predicted_Z[:, 0:1, :, :],sig_predicted_Z[:,1:2,:,:]], 2)
        #predicted_syndrome_X = toric_code.generate_syndrome_X_differentiable(sig_predicted_Z).to(device)
        #predicted_syndrome_Z = toric_code.generate_syndrome_Z_differentiable(sig_predicted_X).to(device)

        #predicted_syndrome_X = torch.flatten(predicted_syndrome_X)
        syndX = torch.flatten(syndX)
        predicted_X = torch.flatten(predicted_X)
        errors_X = torch.flatten(errors_X)
        predicted_Z = torch.flatten(predicted_Z)
        errors_Z = torch.flatten(errors_Z)
        #predicted_syndrome_Z = torch.flatten(predicted_syndrome_Z)
        syndZ = torch.flatten(syndZ)

        loss_errors_X = errors_criterion(predicted_X, errors_X)
        loss_errors_Z = errors_criterion(predicted_Z, errors_Z)
        #with torch.no_grad():
            #loss_syndrome_X = syndrome_criterion(predicted_syndrome_X, syndX)
            #loss_syndrome_Z = syndrome_criterion(predicted_syndrome_Z, syndZ)

        loss_list[0] += loss_errors_Z.item()
        #loss_list[1] += loss_syndrome_X.item()
        loss_list[2] += loss_errors_X.item()
        #loss_list[3] += loss_syndrome_Z.item()

    loss_list[0] /= len(val_loader)
    loss_list[1] /= len(val_loader)
    loss_list[2] /= len(val_loader)
    loss_list[3] /= len(val_loader)
    return loss_list
#[0:errorZ, 1:syndX, 2:errorX, 3:syndZ]

def dataload(path):
    dataset = torch.load(path)
    return dataset

def fig_save(data_a,data_b, file_name):
    x = range(len(data_a))
    fig1 = plt.figure()
    plt.title("loss_graph")
    plt.xlabel("epoch")
    plt.plot(x,data_a)
    plt.plot(x,data_b)
    plt.savefig(file_name)

if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    code_distance = hp.code_distance
    num_data = hp.num_data
    file_path = f'/home/mukailab/test/ogura_code/MyFiles/datasets/dataset_{num_data}_size{code_distance}.pt'
    #preprocess()

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, help="execute mode", default="train", choices=["train", "eval"])
    parser.add_argument("-m", "--model", type=str, help="model path", default=None)
    parser.add_argument("--optuna", type=bool, help="optuna mode", default=False)
    parser.add_argument("--output-model-arch", type=bool, help="output model architecture for tensorboard", default=False)
    args = parser.parse_args()
    """
    #torch.backends.cudnn.benchmark = True
    model = CNN(
        size = hp.code_distance,
        kernel_size = cp.kernel_size,
        resnet_layers= cp.resnet_layers,
        channels= cp.channels,
        activation=cp.activation,
        batch_norm=cp.batch_norm,
        dropout=cp.dp,
        padding=cp.padding,
        nhead=cp.nhead,
        pe_auto=cp.pe_auto,
        weight_init=cp.weight_init
    ).to(device="cuda")
    dataset = torch.load(file_path)
    train_size = int(len(dataset) * cp.dataset_size_train)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = data.random_split(dataset, [train_size, val_size])
    
    train_loader = data.DataLoader(
    train_dataset,
    batch_size = cp.batch_size,
    shuffle = False,
    num_workers = 8
    )
    
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=cp.batch_size,
        shuffle=False,
        num_workers = 8
    )

    #dataset = ToricCodeDataset(dataset, code_distance=hp.code_distance)
    optimizer = torch_optimizer.RAdam(
        params = model.parameters(),
        lr = cp.lr,
        betas = cp.betas)

    param = toric_param(p = hp.error_rate, size = hp.code_distance)
    toric_code = TCT(param)
    errors_X_train_loss = []
    errors_Z_train_loss = []
    errors_X_eval_loss = []
    errors_Z_eval_loss = []
    for i in range(cp.n_epoch):
        train_loss = train(model,
                           train_loader,
                           toric_code=toric_code,
                           optimizer=optimizer)
        
        evaluate_loss = evaluate(model,
                                val_loader,
                                toric_code = toric_code)
        
        print(train_loss)
        print(evaluate_loss)

        
        errors_X_train_loss.append(train_loss[2])
        errors_Z_train_loss.append(train_loss[0])
        errors_X_eval_loss.append(evaluate_loss[2])
        errors_Z_eval_loss.append(evaluate_loss[0])
        #[0:errorZ, 1:syndX, 2:errorX, 3:syndZ]
    file_name_errorX_train = f"/home/mukailab/test/ogura_code/MyFiles/fig/loss/errorX_train/code_distance{hp.code_distance}_errorrate{hp.error_rate}_errorsX_{hp.num_data}.jpg"
    file_name_errorZ_train = f"/home/mukailab/test/ogura_code/MyFiles/fig/loss/errorZ_train/code_distance{hp.code_distance}_errorrate{hp.error_rate}_errorsZ_{hp.num_data}.jpg"
    file_name_errorX_eval = f"/home/mukailab/test/ogura_code/MyFiles/fig/loss/errorX_eval/code_distance{hp.code_distance}_errorrate{hp.error_rate}_syndX_{hp.num_data}.jpg"
    file_name_errorZ_eval = f"/home/mukailab/test/ogura_code/MyFiles/fig/loss/errorZ_eval/code_distance{hp.code_distance}_errorrate{hp.error_rate}_syndZ{hp.num_data}.jpg"
    fig_save(data_a=errors_X_train_loss,data_b=errors_X_eval_loss, file_name=file_name_errorX_train)
    fig_save(data_a=errors_Z_train_loss,data_b=errors_Z_eval_loss, file_name=file_name_errorZ_train)
    #fig_save(errors_X_eval_loss, file_name=file_name_errorX_eval)
    #fig_save(errors_Z_eval_loss, file_name=file_name_errorZ_eval)
    t = datetime.today()
    torch.save(model.cpu().state_dict(), f"/home/mukailab/test/ogura_code/MyFiles/models/{t}.model")

