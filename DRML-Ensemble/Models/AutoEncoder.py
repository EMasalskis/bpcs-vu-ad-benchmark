import dgl
import dgl.nn.pytorch.conv.graphconv as graphconv
import math
import torch
import torch.nn as nn
from torch.nn import init
import Models.HGTLayer as HGTLayer
import Models.HGCNLayer as HGCNLayer
import Models.GCNLayer as GCNLayer
import Models.HGALayer as HGALayer
import Models.util as util


class AutoEncoder(nn.Module):
    def __init__(self, h_feats):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(h_feats, h_feats // 2),
            nn.Linear(h_feats // 2, h_feats // 4),
            nn.Linear(h_feats // 4, 160),
        )


    def forward(self, feature):
        return  self.encoder(feature)