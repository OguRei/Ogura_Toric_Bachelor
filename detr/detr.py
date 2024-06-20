import torch
import numpy as np
import copy

from scipy import optimize as scipy_optim
from torch import nn 
from typing import Optional, Tuple

import os
import sys
sys.path.append(os.path.abspath("../"))
from activations import Mish, Activation
from misc import PositionalEncoding

class SkipConnection(nn.Module):
    def __init__(self, *module_list):
        super(SkipConnection, self).__init__()
        self.module_list = nn.Sequential(*module_list)

    def forward(self, input):
        return input + self.module_list(input)

class TransfomrerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int=2048,
                 dropout: float=0.1,
                 activation: str="relu"):
        super(TransfomrerEncoderLayer, self).__init__()
        self.activation = Activation(activation=activation)
        self.self_attn = nn.MultiheadAttention(
            embed_dim = d_model,
            num_heads = nhead,
            dropout=dropout
        )
        self.linear = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=dim_feedforward),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=dim_feedforward, out_features=d_model),
            nn.Dropout(p=dropout))
        self.dropout = nn.Dropout(p=dropout)
        self.norm1 = nn.LayerNorm(normalized_shape=d_model)
        self.norm2 = nn.LayerNorm(normalized_shape=d_model)

    #situmon
    def forward(self, src: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
        x = self.self_attn(query=src + pe, key=src + pe, value=src)[0]
        x = src + self.dropout(x)
        x = self.norm1(x)
        x = self.linear(x) + x
        x = self.norm2(x)
        return x
    

class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int=2048,
                 dropout: float=0.1,
                 activation: str = "relu"):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim = d_model,
            num_heads=nhead,
            dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.norm1 = nn.LayerNorm(normalized_shape=d_model)
        self.norm2 = nn.LayerNorm(normalized_shape=d_model)
        self.norm3 = nn.LayerNorm(normalized_shape=d_model)
        self.linear = nn.Sequential(
                nn.Linear(in_features=d_model, out_features=dim_feedforward),
                Activation(activation=activation),
                nn.Dropout(p=dropout),
                nn.Linear(in_features=dim_feedforward, out_features=d_model),
                nn.Dropout(p=dropout))
    #situmon
    def forward(self,
                tgt: torch.Tensor,
                pe_dec: torch.Tensor,
                memory: torch.Tensor,
                pe_enc: torch.Tensor) -> torch.Tensor:
        x = self.self_attn(query=tgt + pe_dec, key=tgt + pe_dec, value=tgt)[0]
        x = tgt + self.dropout(x)
        x = self.norm1(x)
        y = self.multihead_attn(query=x + pe_dec, key=memory + pe_enc, value=memory)[0]
        x = x + self.dropout(y)
        x = self.norm2(x)
        x = self.linear(x) + x
        x = self.norm3(x)
        return x
    
class Detr(nn.Module):
    def __init__(self,
                 width: int,
                 height: int,
                 in_channels: int,
                 d_model: int,
                 dim_feedforward: int,
                 nhead: int,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 max_objects: int,
                 num_classes: int,
                 activation: str,
                 sort_box_coord: bool = False,
                 use_auxiliary_loss: bool = False,
                 variable_object_query_min_max: Tuple[int, int] = (1, 1),
                 pe_size: int = 50,
                 pe_auto: bool = False):
        super(Detr, self).__init__()
        self.width = width
        self.height = height
        self.sort_box_coord = sort_box_coord
        self.use_auxiliary_loss = use_auxiliary_loss
        self.variable_object_query_min_max = variable_object_query_min_max
        self.proj = nn.Conv2d(in_channels=in_channels, out_channels=d_model, kernel_size=1)
        encoder_layer = TransfomrerEncoderLayer(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            nhead=nhead,
            activation=activation
        )
        decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            nhead=nhead,
            activation=activation
        )
        self.transfromer_decoder = nn.ModuleList([
            copy.deepcopy(decoder_layer) for _ in range(num_decoder_layers)
            ])
        min, max = variable_object_query_min_max
        self.object_query = nn.Parameter(torch.randn(max_objects,max - min + 1, d_model))
        self.pe = PositionalEncoding(pe_size=pe_size, ndim=d_model, auto=pe_auto)
        self.bounding_box = nn.Sequential(
            SkipConnection(
                nn.Linear(
                    in_features=d_model,
                    out_features=d_model),
                Mish()),
            SkipConnection(
                nn.Linear(
                    in_features=d_model,
                    out_features=d_model),
                Mish()),
            nn.Linear(
                in_features=d_model,
                out_features=4))
        self.cls = nn.Sequential(
            SkipConnection(
                nn.Linear(in_features=d_model, out_features=d_model),
                Mish()),
            nn.Linear(in_features=d_model, out_features=num_classes))
        
    def forward(self, input: torch.Tensor, object_query_index: Optional[torch.Tensor] = None):
        x = self.proj(input)
        B, _, H, W = x.shape
        pos_encode = self.pe(shape=(H, W)).flatten(start_dim=2).permute(2, 0, 1)
        x = x.flatten(start_dim=2).permute(2, 0, 1)
        for mod in self.transformer_encoder:
            x = mod(src=x, pe=pos_encode)

        if object_query_index is not None:
            min, max = self.variable_object_query_min_max
            object_query_index[object_query_index > max] = max
            object_query = self.object_query[:, object_query_index - min, :]
        else:
            object_query = self.object_query.expand(-1, B, -1)
        y = object_query
        outputs = []
        for mod in self.transformer_decoder:
            y = mod(tgt=y, pe_dec=object_query, memory=x, pe_enc=pos_encode)
            output = y.transpose(0, 1)
            if self.use_auxiliary_loss:
                outputs.append(output)
        if self.use_auxiliary_loss:
            outputs = torch.cat(outputs)
        else:
            outputs = output
        bounding_box: torch.Tensor = self.bounding_box(outputs).sin().add(1) / 2.
        if self.sort_box_coord:
            u_, v_ = bounding_box.split(split_size=2, dim=-1)
            mask = (u_[:, :, 0] < v_[:, :, 0]).logical_or(
                (u_[:, :, 0] == v_[:, :, 0]).logical_and(u_[:, :, 1] < v_[:, :, 1])).float().unsqueeze(-1)
            u = mask * u_ + (1 - mask) * v_
            v = (1 - mask) * u_ + mask * v_
            bounding_box = torch.cat((u, v), dim=2)
        cls = self.cls(outputs)
        if self.use_auxiliary_loss:
            return cls.split(B), bounding_box.split(B)
        else:
            return (cls, ), (bounding_box, )

class HungarianLoss(nn.Module):
    def __init__(self,
                 class_weight: torch.Tensor,
                 cls_cost_param: float = 1.,
                 box_cost_param: float = 5.,
                 topk: int = 0):
        super(HungarianLoss, self).__init__()
        self.cls_cost_param = cls_cost_param
        self.box_cost_param = box_cost_param

        self.cls_criterion = nn.NLLLoss(weight=class_weight)
        self.box_criterion = nn.L1Loss()
        assert topk >= 0
        self.k = topk

    def forward(self,
                input_cls: torch.Tensor,
                input_box: torch.Tensor,
                target_cls: torch.Tensor,
                target_box: torch.Tensor,
                nobj: torch.Tensor):
        with torch.no_grad():
            # cls loss
            input_cls_prob_with_empty_0 = input_cls.softmax(dim=2)
            input_cls_prob_with_empty_0[:, :, 0] = 0.

            cls_cost = -torch.cat(
                [torch.index_select(icls, 1, idx).unsqueeze(0)
                    for icls, idx in zip(input_cls_prob_with_empty_0, target_cls)])
            # box loss
            box_cost = torch.cdist(input_box, target_box, p=1)

            cost = self.cls_cost_param * cls_cost + \
                   self.box_cost_param * box_cost

            cost = cost.cpu().detach().numpy()
            cost = np.nan_to_num(cost, nan=0.)
            indices = []
            for c in cost:
                input_indices, target_indices = scipy_optim.linear_sum_assignment(c)
                _indices = sorted(list(zip(input_indices, target_indices)), key=lambda x: x[1])
                input_indices = [index[0] for index in _indices]
                target_indices = [index[1] for index in _indices]
                indices.append((torch.tensor(input_indices), torch.tensor(target_indices)))

        cls_loss = []
        for _input_cls, _target_cls, (input_idx, target_idx) \
                in zip(input_cls.log_softmax(dim=2), target_cls, indices):  # foreach batch
            cls_loss += [
                self.cls_criterion(
                    _input_cls[input_idx],
                    _target_cls[target_idx])]
        cls_loss = torch.stack(cls_loss)

        box_loss = []
        for _input_box, _target_box, (input_idx, target_idx), _nobj \
                in zip(input_box, target_box, indices, nobj):  # foreach batch
            box_loss += [
                self.box_criterion(
                    _input_box[input_idx[:_nobj]],
                    _target_box[target_idx[:_nobj]])]
        box_loss = torch.stack(box_loss)
        if self.k == 0:
            return self.cls_cost_param * cls_loss.mean(), self.box_cost_param * box_loss.mean()
        else:
            loss = self.cls_cost_param * cls_loss + self.box_cost_param * box_loss
            indices = torch.topk(loss, k=self.k).indices
            return self.cls_cost_param * cls_loss[indices].mean(), self.box_cost_param * box_loss[indices].mean()






