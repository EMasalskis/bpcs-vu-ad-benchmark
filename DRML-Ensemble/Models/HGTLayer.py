import dgl
from dgl.nn.pytorch.conv.hgtconv import HGTConv
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from torch.nn import init
import Models.util as util
from dgl.nn.functional import edge_softmax


class HGTAttention(nn.Module):
    def __init__(self, src_dim, dst_dim, hid_dim, out_dim, edge_type, n_heads, use_norm=False):
        super(HGTAttention, self).__init__()
        self.models = nn.ModuleList()
        self.nheads = n_heads
        self.src_dict, self.edge_dict, self.dst_dict = edge_type

        for i in range(n_heads):
            self.models.append(HGTBlock(src_dim, dst_dim, hid_dim, out_dim, edge_type))

    def forward(self, G, src_inp_key, dst_inp_key, out_key, alphe=1):
        feature = []
        for i in range(self.nheads):
            self.models[i](G, src_inp_key, dst_inp_key, f"{out_key}{i}", alphe)
            feature.append(G.nodes[self.dst_dict].data[f"{out_key}{i}"])
        G.nodes[self.dst_dict].data[out_key] = torch.mean(torch.stack(feature, dim=1), dim=1)


class HGTBlock(nn.Module):
    def __init__(self, src_dim, dst_dim, hid_dim, out_dim, edge_type, use_norm=False):
        super(HGTBlock, self).__init__()
        self.out_dim = out_dim
        self.edge_type = edge_type
        self.use_norm = use_norm
        self.hid_dim = hid_dim
        self.sqrt_dk = math.sqrt(hid_dim)

        self.src_t_linear = nn.Sequential(
            nn.Linear(src_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
        )

        self.dst_t_linear = nn.Sequential(
            nn.Linear(dst_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
        )

        self.k_linear = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
        )
        self.v_linear = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
        )
        self.q_linear = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
        )

        self.a_linear = nn.Sequential(
            nn.Linear(hid_dim, out_dim),
        )

        self.skip = nn.Parameter(torch.zeros(1))

        if use_norm:
            self.norms = nn.LayerNorm(out_dim)

    def edge_attention(self, edges):
        m = torch.sum((edges.dst['q'] * edges.src['k']), dim=-1)
        m /= self.sqrt_dk
        m = torch.sigmoid(m)

        return {'a': m, 'v': edges.src['v']}

    def message_func(self, edges):
        return {'v': edges.data['v'], 'a': edges.data['a']}

    def reduce_func(self, nodes):
        # print(nodes.mailbox['a'].shape)
        att = torch.softmax(nodes.mailbox['a'], dim=1)
        h = torch.sum(att.unsqueeze(-1) * nodes.mailbox['v'], dim=1)

        return {'t': h}

    def forward(self, G, src_inp_key, dst_inp_key, out_key, alphe=1):
        src_dict, edge_dict, dst_dict = self.edge_type

        G.nodes[src_dict].data['hid'] = self.src_t_linear(G.nodes[src_dict].data[src_inp_key])
        G.nodes[dst_dict].data['hid'] = self.dst_t_linear(G.nodes[dst_dict].data[dst_inp_key])
        #
        G.nodes[src_dict].data['k'] = self.k_linear(G.nodes[src_dict].data['hid'])
        G.nodes[src_dict].data['v'] = self.v_linear(G.nodes[src_dict].data['hid'])
        G.nodes[dst_dict].data['q'] = self.q_linear(G.nodes[dst_dict].data['hid'])

        G.apply_edges(func=self.edge_attention, etype=edge_dict)
        G.multi_update_all({edge_dict: (self.message_func, self.reduce_func)}, cross_reducer='sum')
        # alphe = torch.sigmoid(self.skip)
        # trans_out = alphe * G.nodes[dst_dict].data['t'] + (1 - alphe) * G.nodes[dst_dict].data['hid']

        trans_out = G.nodes[dst_dict].data['t'] + alphe * G.nodes[dst_dict].data['hid']
        trans_out = self.a_linear(trans_out)
        if self.use_norm:
            G.nodes[dst_dict].data[out_key] = self.norms(trans_out)
        else:
            G.nodes[dst_dict].data[out_key] = trans_out

    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_types={}, num_types={})'.format(
            self.__class__.__name__, self.in_dim, self.out_dim,
            self.num_types, self.num_relations)
