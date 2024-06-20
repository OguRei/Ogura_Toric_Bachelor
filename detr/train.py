import time
import sys
import os
import argparse
import copy
import random

import numpy as np
from numpy.core.fromnumeric import size
import torch
import torch_optimizer
from torch import nn, optim, cuda, backends
from torch.nn import functional as F
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm, trange
from typing import List, Tuple
from datetime import datetime

sys.path.append(os.path.abspath("../"))
from toric_code_ogura import ToricCode
from utils import \
        load_model, \
        preprocess, \
        ToricCodeDatasetRuntime, \
        ToricCodeDataset, \
        init_weight, \
        init_weights, \
        visualize_syndrome
from detr import Detr, HungarianLoss
from cnn.cnn import CNN
from cnn.cnnparam import CnnParam as cp
from mwpm.MWPMdecoder import decode as MWPM
from hyperparam import HyperParam as hp


def x1y1x2y2_to_cxcywh(box: torch.Tensor) -> torch.Tensor:
    '''box: [B, L, 4]'''
    cx = (box[:, :, 0] + box[:, :, 2]) / 2
    cy = (box[:, :, 1] + box[:, :, 3]) / 2
    w = (box[:, :, 0] - box[:, :, 2]).abs()
    h = (box[:, :, 1] - box[:, :, 3]).abs()
    return torch.stack((cx, cy, w, h), dim=2)


def cxcywh_to_x1y1x2y2(box: torch.Tensor, eval: bool = False) -> torch.Tensor:
    x1 = box[:, :, 0] - box[:, :, 2] / 2
    y1 = box[:, :, 1] - box[:, :, 3] / 2
    x2 = box[:, :, 0] + box[:, :, 2] / 2
    y2 = box[:, :, 1] + box[:, :, 3] / 2
    box = torch.stack((x1, y1, x2, y2), dim=2)
    if eval:
        box = box.clip(min=0., max=1.)
    return box


def train(detr: Detr,
          backbone: CNN,
          train_loader: data.DataLoader,
          optimizerD: optim.Optimizer,
          optimizerB: optim.Optimizer,
          device="cuda") -> List[float]:
    criterion = HungarianLoss(
        class_weight=torch.tensor(hp.class_weight, device=device),
        cls_cost_param=hp.cls_param,
        box_cost_param=hp.box_param,
        topk=hp.topk)
    detr.train()
    def freeze_batchnorm(m: nn.Module):
        if type(m) is nn.BatchNorm2d:
            m.requires_grad_(False)
    backbone.train()
    if hp.freeze_batchnorm:
        backbone.apply(freeze_batchnorm)

    train_cls_loss = 0.0
    train_box_loss = 0.0
    for syndrome, cls, coord, nobj in tqdm(train_loader):
        syndrome = syndrome.to(device)
        cls = cls.to(device)
        coord = coord.to(device)
        if hp.cxcywh:
            coord = x1y1x2y2_to_cxcywh(coord)

        cls_loss = torch.zeros([1], device=device)
        box_loss = torch.zeros([1], device=hp.device)

        optimizerB.zero_grad(set_to_none=True)
        optimizerD.zero_grad(set_to_none=True)
        with cuda.amp.autocast(hp.use_fp16):
            x = backbone.features(syndrome)
            num_syndrome = syndrome.round().int().sum(dim=(1, 2, 3))
            predicted_cls, predicted_box = detr(x, object_query_index=torch.div(num_syndrome, 2, rounding_mode="trunc"))
            for _predicted_cls, _predicted_box in zip(predicted_cls, predicted_box):
                _cls_loss, _box_loss = criterion(
                        _predicted_cls, _predicted_box, cls, coord, nobj)
                cls_loss += _cls_loss
                box_loss += _box_loss
            cls_loss /= len(predicted_cls)
            box_loss /= len(predicted_box)
            loss = cls_loss + box_loss
            if hp.use_fp16:
                scaler.scale(loss).backward()
                scaler.step(optimizerD)
                scaler.step(optimizerB)
                scaler.update()
            else:
                loss.backward()
                optimizerD.step()
                optimizerB.step()


        train_cls_loss += cls_loss.item()
        train_box_loss += box_loss.item()

    lr_schedulerB.step()
    lr_schedulerD.step()

    train_cls_loss /= len(train_loader)
    train_box_loss /= len(train_loader)
    return train_cls_loss, train_box_loss

def evaluate_as_detr(detr: Detr,
             backbone: CNN,
             val_loader: data.DataLoader,
             device: str="cuda") -> List[float]:
    criterion = HungarianLoss(
        class_weight=torch.tensor(hp.class_weight, device=device),
        cls_cost_param=hp.cls_param,
        box_cost_param=hp.box_param,
        topk=0)
    detr.eval()
    backbone.eval()

    eval_cls_loss = 0.0
    eval_box_loss = 0.0
    with torch.no_grad():
        for syndrome, cls, coord, nobj in tqdm(val_loader):
            syndrome = syndrome.to(device)
            cls = cls.to(device)
            coord = coord.to(device)
            if hp.cxcywh:
                coord = x1y1x2y2_to_cxcywh(coord)

            cls_loss = torch.zeros([1], device=device)
            box_loss = torch.zeros([1], device=device)

            with cuda.amp.autocast(hp.use_fp16):
                x = backbone.features(syndrome)
                num_syndrome = syndrome.round().int().sum(dim=(1, 2, 3))
                predicted_cls, predicted_box = detr(x, object_query_index=torch.div(num_syndrome, 2, rounding_mode="trunc"))
                for _predicted_cls, _predicted_box in zip(predicted_cls, predicted_box):
                    _cls_loss, _box_loss = criterion(
                            _predicted_cls, _predicted_box, cls, coord, nobj)
                    cls_loss += _cls_loss
                    box_loss += _box_loss
                cls_loss /= len(predicted_cls)
                box_loss /= len(predicted_box)
                loss = cls_loss + box_loss

            eval_cls_loss += cls_loss.item()
            eval_box_loss += box_loss.item()
        eval_cls_loss /= len(val_loader)
        eval_box_loss /= len(val_loader)
        return eval_cls_loss, eval_box_loss


def evaluate(detr: Detr,
           backbone: CNN,
           n_iter: int,
           device: str="cpu"):
    toric_code = ToricCode().to(device)
    detr.eval()
    backbone.eval()
    def dist_min(coord: List[Tuple[int, int]], u: Tuple[int, int]):
        '''coord: x1y1x2y2'''
        round_u = tuple(map(lambda x: round(x), u))
        if round_u in coord:
            return coord.index(round_u)
        def dist(a: Tuple[int, int], b: Tuple[int, int], L: int):
            _x_dist = np.abs(a[0] - b[0])
            _y_dist = np.abs(a[1] - b[1])
            return min(_x_dist, L - _x_dist) + min(_y_dist, L - _y_dist)
        dist_list = list(map(lambda x: dist(x, u, hp.L), coord))
        return dist_list.index(min(dist_list))
        # return np.argmin(dist_list)

    not_correct = 0
    spent = 0.0
    max_spent = 0.0
    count_n_errors = [0] * 2 * hp.L ** 2
    count_n_syndrome = [0] * hp.L ** 2
    count_n_errors_not_correct = [0] * 2 * hp.L ** 2
    count_n_syndrome_not_correct = [0] * hp.L ** 2
    with torch.no_grad():
        for _ in trange(n_iter):
            matching = []
            errors = toric_code.generate_errors(
                    torch.full((1, 2, hp.L, hp.L), fill_value=False),
                    hp.error_rate)
            errors_ = errors.clone()
            syndrome = (toric_code.syndrome(errors.to(device).float()) > 0.5).float().cpu()
            n_errors = errors.sum().int().item()
            count_n_errors[n_errors] += 1
            n_syndrome = syndrome.sum().int().item()
            count_n_syndrome[n_syndrome] += 1

            coord = list(zip(*np.where((syndrome == 1.0).cpu())[2:4]))
            coord_ = copy.deepcopy(coord)
            before = time.perf_counter()
            # ---------------------------------------------------
            num_matching = len(coord) // 2
            if num_matching > hp.max_objects:
                print("num_matching > max_objects")
                continue
            x = backbone.features(syndrome.cuda(non_blocking=True))
            num_syndrome = syndrome.round().int().sum(dim=(1, 2, 3))
            (predicted_cls, ), (predicted_box, ) = detr(x, object_query_index=torch.div(num_syndrome, 2, rounding_mode="trunc"))
            predicted_cls = predicted_cls[0].sigmoid().cpu()
            if hp.cxcywh:
                predicted_box = cxcywh_to_x1y1x2y2(predicted_box, eval=True)
            predicted_box = predicted_box[0].mul(hp.L)
            predicted = zip(predicted_cls, predicted_box)
            predicted = sorted(predicted, key=lambda x: x[0][1], reverse=True)
            for _predicted_cls, _predicted_box in predicted[:num_matching]:
                u0, u1, v0, v1 = _predicted_box.tolist()
                u, v = (u0, u1), (v0, v1)
                idx = dist_min(coord, u)
                u = coord.pop(idx)
                idx = dist_min(coord, v)
                v = coord.pop(idx)
                matching.append((u, v))
            # ---------------------------------------------------
            elapsed_time = time.perf_counter() - before
            spent += elapsed_time
            max_spent = max(elapsed_time, max_spent)
            for u, v in matching:
                errors = ToricCode.decode(errors, u, v, hp.L)
            syndrome_ = (toric_code.syndrome(errors.to(device)) > 0.5).float()
            if (syndrome_ == 0.0).all() and toric_code.not_has_non_trivial(errors):
                pass
            else:
                not_correct += 1
                matching = MWPM(coord_)
                for u, v in matching:
                    errors_ = ToricCode.decode(errors_, u, v, hp.L)
                if (toric_code.syndrome(errors_) == 0.0).all() and \
                        toric_code.not_has_non_trivial(errors_):
                    visualize_syndrome(syndrome[0][0])
                    count_n_errors_not_correct[n_errors] += 1
                    count_n_syndrome_not_correct[n_syndrome] += 1
                else:
                    print("Failed to decode, because of too match errors")


    print("Errors:")
    for n, (e, c) in enumerate(zip(count_n_errors, count_n_errors_not_correct)):
        print(f"{n}: {c} / {e} == {0.0 if e == 0 else c / e}")
    print("Syndrome:")
    for n, (s, c) in enumerate(zip(count_n_syndrome[::2], count_n_syndrome_not_correct[::2])):
        print(f"{n}: {c} / {s} == {0.0 if s == 0 else c / s}")

    print("mean:", spent / n_iter)
    print("max:", max_spent)
    print(f"error rate: {not_correct} / {n_iter} = {not_correct / n_iter}")


if __name__ == "__main__":
    if not hp.dataset_runtime:
        preprocess()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, help="execute mode", default="train", choices=["train", "eval"])
    parser.add_argument("-m", "--model", type=str, help="model path", default=None)
    parser.add_argument("-pb", "--pretrained_backbone", type=str, help="pretrained backbone path", default=None)
    args = parser.parse_args()
    backends.cudnn.enabled = True
    backends.cudnn.benchmark = True
    if args.mode == "train":
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
    else:
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

    if hp.dataset_runtime:
        dataset = ToricCodeDatasetRuntime(
                length=hp.dataset_size,
                size=hp.L,
                max_objects=hp.max_objects,
                error_rate=hp.error_rate,
                dataset_algo=hp.dataset_algo)
        val_size = len(dataset) // 20
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = data.random_split(dataset, [train_size, val_size])
    else:
        train_dataset = ToricCodeDataset(
            size=hp.L,
            mode="match",
            max_objects=hp.max_objects)
        val_size = hp.dataset_size // 20
        val_dataset = ToricCodeDatasetRuntime(
                length=val_size,
                size=hp.L,
                max_objects=hp.max_objects,
                error_rate=hp.error_rate,
                dataset_algo="rate")
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=hp.batch_size,
        shuffle=True,
        num_workers=8 if hp.env == "lab" else 16)
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=hp.batch_size,
        shuffle=True,
        num_workers=8)
    backbone = CNN(
        size=hp.L,
        kernel_size=cp.kernel_size,
        resnet_layers=cp.resnet_layers,
        channels=cp.channels,
        activation=cp.activation,
        activation_args=cp.activation_args,
        weight_init=cp.weight_init,
        dropout=hp.dp_backbone,
        nhead=cp.nhead,
        batch_norm=cp.batch_norm,
        padding=cp.padding,
        pe_auto=cp.pe_auto).to(hp.device)
    detr = Detr(
        width=hp.L, height=hp.L,
        in_channels=hp.feature_size,
        d_model=hp.d_model,
        nhead=hp.nhead,
        dim_feedforward=hp.dim_feedforward,
        num_encoder_layers=hp.num_encoder_layers,
        num_decoder_layers=hp.num_decoder_layers,
        max_objects=hp.max_objects,
        num_classes=hp.num_classes,
        activation=hp.activation,
        use_auxiliary_loss=hp.use_auxiliary_loss,
        sort_box_coord=hp.sort_box_coord,
        variable_object_query_min_max=hp.variable_object_query_min_max,
        pe_size=hp.pe_size,
        pe_auto=hp.pe_auto).to(hp.device)
    detr.apply(init_weights)

    if hp.optimizer == "adamw":
        optimizerB = optim.AdamW(
                backbone.parameters(),
                lr=hp.lr_backbone,
                weight_decay=hp.weight_decay,
                eps=hp.eps)
        optimizerD = optim.AdamW(
                detr.parameters(),
                lr=hp.lr,
                weight_decay=hp.weight_decay,
                eps=hp.eps)
    elif hp.optimizer == "adablief":
        optimizerB = torch_optimizer.AdaBelief(
                backbone.parameters(),
                lr=hp.lr_backbone,
                weight_decay=hp.weight_decay,
                eps=hp.eps)
        optimizerD = torch_optimizer.AdaBelief(
                detr.parameters(),
                lr=hp.lr,
                weight_decay=hp.weight_decay,
                eps=hp.eps)
    elif hp.optimizer == "radam":
        optimizerB = torch_optimizer.RAdam(
                backbone.parameters(),
                lr=hp.lr_backbone,
                weight_decay=hp.weight_decay,
                eps=hp.eps)
        optimizerD = torch_optimizer.RAdam(
                detr.parameters(),
                lr=hp.lr,
                weight_decay=hp.weight_decay,
                eps=hp.eps)
    else:
        assert False

    lr_schedulerB = optim.lr_scheduler.StepLR(optimizerB, step_size=hp.lr_step, gamma=0.1)
    lr_schedulerD = optim.lr_scheduler.StepLR(optimizerD, step_size=hp.lr_step, gamma=0.1)

    nn.utils.clip_grad_norm_(backbone.parameters(), max_norm=1.0)
    nn.utils.clip_grad_norm_(detr.parameters(), max_norm=1.0)
    if hp.use_fp16:
        scaler = cuda.amp.GradScaler()
    if args.mode == "train":
        writer = SummaryWriter(log_dir="./log")
        if hp.use_pretrained:
            backbone = load_model(backbone, path=args.pretrained_backbone, device=hp.device)
        global_step = 0

        for epoch in trange(hp.num_epoch, desc="epoch"):
            train_cls_loss, train_box_loss = train(detr, backbone, train_loader, optimizerD, optimizerB, hp.device)
            if epoch % hp.save_per == 0:
                val_cls_loss, val_box_loss = evaluate_as_detr(detr, backbone, val_loader, hp.device)
                tag = f"{hp.cls_param} {hp.box_param} {hp.class_weight} {hp.dataset_size} {hp.num_encoder_layers} {hp.num_decoder_layers} {cp.resnet_layers} {hp.lr} {hp.lr_backbone} {hp.use_pretrained} {hp.use_auxiliary_loss} {cp.activation} {hp.activation} {hp.nhead} {hp.norm} {hp.pe_size} {hp.optimizer} {cp.dataset_runtime} {hp.dataset_runtime} {hp.batch_size} {hp.lr_step} {hp.max_objects} {hp.freeze_batchnorm} {cp.pe_auto} {cp.dataset_algo} {hp.topk} {hp.variable_object_query_min_max} sin bounding_box"
                writer.add_scalars("class loss", {
                    f"train {tag}": train_cls_loss,
                    f"val {tag}": val_cls_loss,
                    }, global_step)
                writer.add_scalars("box loss", {
                    f"train {tag}": train_box_loss,
                    f"val {tag}": val_box_loss,
                    }, global_step)
                t = datetime.today()
                torch.save({
                    "backbone": backbone.state_dict(),
                    "detr": detr.state_dict()
                    }, os.path.join(hp.model_dir, f"{t}.model"))
            global_step += 1
    elif args.mode == "eval":
        path = args.model
        load_model(detr, key="detr", path=path, device=hp.device)
        load_model(backbone, key="backbone", path=path, device=hp.device)
        detr.use_auxiliary_loss = False
        evaluate(detr, backbone, 100000)

