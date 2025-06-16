import dgl
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from torch.nn import init


class NodeLevelAttentionBlock(nn.Module):
    def __init__(self, src_dim, dst_dim, hid_dim, out_dim, edge_type, dropout=0.2, use_norm=False):
        super(NodeLevelAttentionBlock, self).__init__()
        self.edge_type = edge_type
        self.hid_dim = hid_dim
        self.use_norm = use_norm
        self.src_t_linear = nn.Sequential(
            nn.Linear(src_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
        )
        self.dst_t_linear = nn.Sequential(
            nn.Linear(dst_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
        )
        # an,hid
        self.AttentionSrc = nn.Sequential(
            nn.Linear(src_dim, hid_dim),
        )
        # bn,hid
        self.AttentionDst = nn.Sequential(
            nn.Linear(dst_dim, hid_dim),
        )
        self.Attention = nn.Sequential(
            nn.Linear(2 * hid_dim, 1),
        )

        self.a_linear = nn.Sequential(
            nn.Linear(hid_dim, out_dim),
        )
        if use_norm:
            self.norms = nn.LayerNorm(out_dim)
        self.skip = nn.Parameter(torch.zeros(1))
        self.l = 0

    def edge_attention(self, edges):
        m = self.Attention(torch.cat((edges.dst['attention'], edges.src['attention']), dim=1))
        return {'a': torch.squeeze(m), 'v': edges.src['hid']}

        # return {'a': torch.squeeze(m), 'v': torch.unsqueeze(edges.src['hid'],dim=-2)}

    def message_func(self, edges):
        return {'v': edges.data['v'], 'a': edges.data['a']}

    def reduce_func(self, nodes):
        # 会把相同边数的节点,作为一个批次
        att = F.softmax(torch.tanh(nodes.mailbox['a']), dim=1)
        # value = torch.repeat_interleave(nodes.mailbox['v'],2,dim=-2)
        # print(value.shape)

        h = torch.sum(att.unsqueeze(dim=-1) * nodes.mailbox['v'], dim=1)
        return {'t': h}

    def forward(self, G, src_inp_key, dst_inp_key, out_key):
        # print(G)
        src_dict, edge_dict, dst_dict = self.edge_type

        G.nodes[src_dict].data['hid'] = self.src_t_linear(G.nodes[src_dict].data[src_inp_key])
        G.nodes[dst_dict].data['hid'] = self.dst_t_linear(G.nodes[dst_dict].data[dst_inp_key])

        G.nodes[src_dict].data['attention'] = self.AttentionSrc(G.nodes[src_dict].data[src_inp_key])
        G.nodes[dst_dict].data['attention'] = self.AttentionDst(G.nodes[dst_dict].data[dst_inp_key])

        G.apply_edges(func=self.edge_attention, etype=edge_dict)
        G.multi_update_all({edge_dict: (self.message_func, self.reduce_func)}, cross_reducer='mean')

        # trans_out = torch.cat((G.nodes[dst_dict].data['t'], G.nodes[dst_dict].data['hid']), dim=1)
        alphe = torch.sigmoid(self.skip)
        trans_out = alphe * G.nodes[dst_dict].data['t'] + (1 - alphe) * G.nodes[dst_dict].data['hid']

        trans_out = self.a_linear(trans_out)

        # trans_out = trans_out + G.nodes[dst_dict].data['hid']

        if self.use_norm:
            G.nodes[dst_dict].data[out_key] = self.norms(trans_out)
        else:
            G.nodes[dst_dict].data[out_key] = trans_out
