import torch
import torch.nn as nn


class EdgeDropout(nn.Module):
    def __init__(self, keep_prob=0.5):
        super(EdgeDropout, self).__init__()
        assert keep_prob > 0
        self.keep_prob = keep_prob
        self.register_buffer("p", torch.tensor(keep_prob))

    def forward(self, edge_index, edge_weight):
        if self.training:
            mask = torch.rand(edge_index[0].shape[0], device=edge_weight.device)
            mask = torch.floor(mask + self.p).type(torch.bool)
            edge_index = (edge_index[0][mask], edge_index[1][mask])
            edge_weight = edge_weight[mask] / self.p
        return edge_index, edge_weight

    def __repr__(self):
        return '{}(keep_prob={})'.format(self.__class__.__name__, self.keep_prob)
