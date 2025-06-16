import dgl
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn



class GCNLayer(nn.Module):
    def __init__(self, in_features, out_feature):
        super().__init__()
        self.conv1 = dgl.nn.GraphConv(in_features, out_feature, allow_zero_in_degree=True)

    def forward(self,block,x):
        x = F.relu(self.conv1(block, x))
        return x