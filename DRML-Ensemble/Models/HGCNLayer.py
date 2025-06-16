import dgl
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from torch.nn import init
import Models.util as util
import dgl.nn.pytorch.conv.graphconv
from dgl.utils import expand_as_pair


class DGLHGCNBlock(nn.Module):
    def __init__(self, type, in_feats, out_feats, src_dim, dst_dim):
        super(DGLHGCNBlock, self).__init__()
        self.edge_type = type
        self.norm = 'both'
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.linear1 = nn.Linear(src_dim, in_feats)
        self.linear2 = nn.Linear(dst_dim, in_feats)
        self.ResLinear = nn.Linear(in_feats, out_feats)

        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        self.bias = nn.Parameter(torch.Tensor(out_feats))
        self.reset_parameters()
        # self.activation = nn.ReLU()
        self.activation = None

    def reset_parameters(self):
        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, G, src_inp_key, dst_inp_key, out_key):
        src_dict, edge_dict, dst_dict = self.edge_type

        feat_src = self.linear1(G.nodes[src_dict].data[src_inp_key])
        feat_dst = self.linear2(G.nodes[dst_dict].data[dst_inp_key])

        degs = G.out_degrees(etype=edge_dict).float().clamp(min=1)
        if self.norm == 'both':
            norm = torch.pow(degs, -0.5)
        else:
            norm = 1.0 / degs
        shp = norm.shape + (1,) * (G.nodes[src_dict].data['h'].dim() - 1)
        norm = torch.reshape(norm, shp)
        feat_src = feat_src * norm

        aggregate_fn = fn.copy_src('hid', 'm')
        if self.in_feats > self.out_feats:
            feat_src = torch.matmul(feat_src, self.weight)
            G.nodes[src_dict]['hid'] = feat_src
            G.multi_update_all({edge_dict: (aggregate_fn, fn.sum(msg='m', out='hhh'))}, cross_reducer='sum')
            rst = G.nodes[dst_dict].data['hhh']
        else:
            # aggregate first then mult W
            G.nodes[src_dict].data['hid'] = feat_src
            G.multi_update_all({edge_dict: (aggregate_fn, fn.sum(msg='m', out='hhh'))}, cross_reducer='sum')
            rst = G.nodes[dst_dict].data['hhh']
            rst = torch.matmul(rst, self.weight)

        if self.norm in ['right', 'both']:
            degs = G.in_degrees(etype=edge_dict).float().clamp(min=1)
            if self.norm == 'both':
                norm = torch.pow(degs, -0.5)
            else:
                norm = 1.0 / degs

            shp = norm.shape + (1,) * (feat_dst.dim() - 1)
            norm = torch.reshape(norm, shp)
            rst = rst * norm

        rst = rst + self.ResLinear(feat_dst)
        if self.bias is not None:
            rst = rst + self.bias

        if self.activation is not None:
            rst = self.activation(rst)
        G.nodes[dst_dict].data[out_key] = rst
        # print(G.nodes[dst_dict].data[out_key].shape)[09/10 16:30:00][INFO] DRHGCN:  218: final results: {'aupr': 0.8419708061432054, 'auroc': 0.9543170135698775, 'lagcn_aupr': 0.8419818950405311, 'lagcn_auc': 0.9543225086933815, 'lagcn_f1_score': 0.7811804, 'lagcn_accuracy': 0.9621132, 'lagcn_recall': 0.74390244, 'lagcn_specificity': 0.9839342, 'lagcn_precision': 0.82239157}