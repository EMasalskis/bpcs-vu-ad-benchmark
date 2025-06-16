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


def init_weights(net, init_type='kaiming', init_gain=math.sqrt(1)):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 1, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm1d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super(MLPPredictor, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(h_feats, h_feats // 2),
            nn.BatchNorm1d(h_feats // 2),
            nn.ReLU(),
            nn.Linear(h_feats // 2, h_feats // 4),
            nn.BatchNorm1d(h_feats // 4),
            nn.ReLU(),
            nn.Linear(h_feats // 4, 1),
            nn.Sigmoid(),
        )

    def reduce_func(self, nodes):
        return None

    def apply_edges(self, edges):
        # h = torch.mean(edges.src['h'] * edges.dst['h'], dim=1)
        # return {'score': torch.sigmoid(h)}
        h = torch.cat((edges.src['h'], edges.dst['h']), 1)
        # h = edges.src['h'] + edges.dst['h']
        return {'score': self.linear(h).squeeze(1)}

    def forward(self, g, h_drug, d_disease):
        with g.local_scope():
            g.nodes['Drug'].data['h'] = h_drug
            g.nodes['Disease'].data['h'] = d_disease
            g.apply_edges(self.apply_edges)
            return g.edges['treath'].data['score']


class ScorePredictor(nn.Module):
    def __init__(self, dim):
        super(ScorePredictor, self).__init__()
        self.w = nn.Parameter(torch.ones(dim, dim))
        init.xavier_uniform_(self.w)

    def apply_edges(self, edges):
        h = torch.mean(edges.src['h'] * edges.dst['h'], dim=1)
        return {'score': torch.sigmoid(h)}

    def forward(self, g, h_drug, d_disease):
        with g.local_scope():
            g.nodes['Drug'].data['h'] = h_drug
            g.nodes['Disease'].data['h'] = d_disease
            g.apply_edges(self.apply_edges)
            return g.edges['treath'].data['score']


class GCNPredictor(nn.Module):
    def __init__(self, dim, device="cuda:0"):
        super(GCNPredictor, self).__init__()
        # init.xavier_uniform_(self.w)
        self.dim = dim
        self.device = device
        self.convGCN = graphconv.GraphConv(self.dim * 2, self.dim * 2, allow_zero_in_degree=True)
        self.scorePredictor = MLPPredictor(dim * 4)

    def forward(self, g, h_drug, d_disease):
        with g.local_scope():
            drug_number = h_drug.shape[0]
            disease_number = d_disease.shape[0]
            g.nodes['Drug'].data['h'] = torch.cat((h_drug, torch.zeros((drug_number, self.dim)).to(self.device)),
                                                  dim=-1)
            g.nodes['Disease'].data['h'] = torch.cat((torch.zeros(disease_number, self.dim).to(self.device), d_disease),
                                                     dim=-1)
            feat = torch.cat((g.nodes['Drug'].data['h'], g.nodes['Disease'].data['h']), dim=0)

            # feat = torch.cat((h_drug, d_disease), dim=0)
            sub_g = dgl.edge_type_subgraph(g, [('Drug', 'treath', 'Disease')])
            # 前面是药物后面是疾病
            homo_g = dgl.to_homogeneous(sub_g)

            feature = self.convGCN(homo_g, feat)
            g.nodes['Drug'].data['h'] = feature[:drug_number]
            g.nodes['Disease'].data['h'] = feature[drug_number:]
            return self.scorePredictor(sub_g, g.nodes['Drug'].data['h'], g.nodes['Disease'].data['h'])


def Norm(feature, eps=1e-5):
    mean = torch.mean(feature, dim=0)
    std = torch.std(feature, dim=0)
    return (feature - mean) / (std + eps)


# 简单相加
class SampleSumModel(nn.Module):
    def __init__(self, G, feature_dim, agg_type="mean", have_feature_in_Drug_and_Disease=False):
        super(SampleSumModel, self).__init__()
        self.DG = ('Disease', 'interaction_DG', 'Protein')
        self.DT = ('Drug', 'interaction_DT', 'Protein')
        self.G = G
        self.agg_type = agg_type
        self.norm = True
        self.have_feature = have_feature_in_Drug_and_Disease
        self.feature_init(have_feature_in_Drug_and_Disease)
        self.predictor = MLPPredictor(G.nodes["Disease"].data["he"].shape[1] + G.nodes["Drug"].data["he"].shape[1])

        # self.GCNLayerDrug = GCNLayer.GCNLayer(G.nodes["Disease"].data["he"].shape[1],
        #                                       G.nodes["Disease"].data["he"].shape[1])
        # self.GCNLayerDisease = GCNLayer.GCNLayer(G.nodes["Drug"].data["he"].shape[1],
        #                                          G.nodes["Drug"].data["he"].shape[1])
        # self.c = 4 if self.have_feature else 2
        # self.predictor = MLPPredictor(feature_dim * self.c)
        # if (self.have_feature):
        #     self.linear_Drug = nn.Sequential(
        #         nn.Linear(G.nodes["Drug"].data["h"].shape[1], feature_dim)
        #     )
        #
        #     self.linear_Disease = nn.Sequential(
        #         nn.Linear(G.nodes["Disease"].data["h"].shape[1], feature_dim)
        #     )

    # 蛋白质 ---> 疾病,药物
    def feature_init(self, have_feature=False):
        if (self.agg_type == "mean"):
            self.G.multi_update_all({"interaction_DG": (self.message_func_protein, self.reduce_func_protein_mean)},
                                    cross_reducer='mean')
            self.G.multi_update_all({'interaction_DT': (self.message_func_protein, self.reduce_func_protein_mean)},
                                    cross_reducer='mean')

        elif (self.agg_type == "sum"):
            self.G.multi_update_all({"interaction_DG": (self.message_func_protein, self.reduce_func_protein_sum)},
                                    cross_reducer='sum')
            self.G.multi_update_all({'interaction_DT': (self.message_func_protein, self.reduce_func_protein_sum)},
                                    cross_reducer='sum')

        if (self.norm):
            self.G.nodes['Drug'].data['he'] = Norm(self.G.nodes['Drug'].data['he'])
            self.G.nodes['Disease'].data['he'] = Norm(self.G.nodes['Disease'].data['he'])
        if (self.have_feature):
            # self.G.nodes['Drug'].data['h'] = self.linear_Drug(self.G.nodes['Drug'].data['h'])
            # self.G.nodes['Disease'].data['h'] = self.linear_Disease(self.G.nodes['Disease'].data['h'])
            self.G.nodes['Drug'].data['he'] = torch.cat(
                [self.G.nodes['Drug'].data['he'], self.G.nodes['Drug'].data['h']],
                dim=1)
            self.G.nodes['Disease'].data['he'] = torch.cat(
                [self.G.nodes['Disease'].data['he'], self.G.nodes['Disease'].data['h']],
                dim=1)

    def reduce_func_protein_mean(self, nodes):
        return {'he': torch.mean(nodes.mailbox['m'], dim=1)}

    def reduce_func_protein_sum(self, nodes):
        return {'he': torch.sum(nodes.mailbox['m'], dim=1)}

    def message_func_protein(self, edges):
        return {'m': edges.src['h']}

    def forward(self, sub_edge):
        with self.G.local_scope():
            # self.G.nodes['Drug'].data['he'] = self.GCNLayerDrug(self.G["interaction_DrugD"],
            #                                                     self.G.nodes["Drug"].data['he'])
            # self.G.nodes['Disease'].data['he'] = self.GCNLayerDisease(self.G["interaction_DisD"],
            #                                                           self.G.nodes["Disease"].data['he'])
            scores = self.predictor(sub_edge, self.G.nodes['Drug'].data['he'], self.G.nodes['Disease'].data['he'])
            return scores


class SimModel(nn.Module):
    def __init__(self, G, feature_dim, have_feature_in_Drug_and_Disease=False):
        super(SimModel, self).__init__()
        self.DG = ('Disease', 'interaction_DG', 'Protein')
        self.DT = ('Drug', 'interaction_DT', 'Protein')
        self.G = G
        self.feature_init(have_feature_in_Drug_and_Disease)
        dim = G.nodes["Disease"].data["he"].shape[1] + G.nodes["Drug"].data["he"].shape[1]

        print(f"DNN_dim = {dim}")
        self.predictor = MLPPredictor(G.nodes["Disease"].data["he"].shape[1] + G.nodes["Drug"].data["he"].shape[1])

    # 蛋白质 ---> 疾病,药物
    def feature_init(self, have_feature=False):
        self.G.nodes['Drug'].data['he'] = self.G.nodes['Drug'].data['h']
        self.G.nodes['Disease'].data['he'] = self.G.nodes['Disease'].data['h']

    def forward(self, sub_edge):
        with self.G.local_scope():
            score = self.predictor(sub_edge, self.G.nodes['Drug'].data['he'], self.G.nodes['Disease'].data['he'])
            return score


class AutoEncoder(nn.Module):
    def __init__(self, h_feats):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(h_feats, h_feats // 2),
            nn.BatchNorm1d(h_feats // 2),
            nn.ReLU(),
            nn.Linear(h_feats // 2, h_feats // 4),
            nn.BatchNorm1d(h_feats // 4),
            nn.ReLU(),
            nn.Linear(h_feats // 4, 160),
        )

    def forward(self, feature):
        return self.encoder(feature)


class ChannelFusionAttention(nn.Module):
    def __init__(self, channelNumber):
        super(ChannelFusionAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(channelNumber, channelNumber),
            nn.ReLU(),
            nn.Linear(channelNumber, channelNumber),
            nn.Sigmoid()
        )

    def forward(self, features):
        features = torch.stack(features, dim=1)
        t, _ = features.max(dim=-1)
        a = self.attention(t)
        a = torch.unsqueeze(a, dim=-1)
        features = a * features
        return features.mean(dim=1)


# Attention
# class FeatureFusionAttention(nn.Module):
#     def __init__(self, channelNumber):
#         super(FeatureFusionAttention, self).__init__()
#         self.attention = nn.Parameter(torch.ones(channelNumber) / channelNumber)
#
#     def forward(self, features):
#         features = torch.stack(features, dim=1)
#         a = torch.softmax(self.attention, 0)
#         a = torch.unsqueeze(a, dim=-1)
#         # print(a.shape)
#         # print(features.shape)
#         features = a * features
#         return features.sum(dim=1)


# Cat
class FeatureFusionAttention(nn.Module):
    def __init__(self, channelNumber):
        super(FeatureFusionAttention, self).__init__()
        self.attention = nn.Parameter(torch.ones(channelNumber) / channelNumber)

    def forward(self, features):
        # features = torch.stack(features, dim=1)
        return torch.cat(features, dim=1)


# 蛋白质融合消融
# MyProteinFusion
class ProteinFusionAttention(nn.Module):
    def __init__(self):
        super(ProteinFusionAttention, self).__init__()

    def forward(self, features):
        features = torch.stack(features, dim=1)
        t, _ = features.max(dim=-1)
        a = torch.softmax(t, 1)
        a = torch.unsqueeze(a, dim=-1)
        features = a * features
        return features.sum(dim=1)


# 相加
# class ProteinFusionAttention(nn.Module):
#     def __init__(self):
#         super(ProteinFusionAttention, self).__init__()
#
#     def forward(self, features):
#         features = torch.stack(features, dim=1)
#         return features.sum(dim=1)

# # 语义级Attention
# class ProteinFusionAttention(nn.Module):
#     def __init__(self):
#         super(ProteinFusionAttention, self).__init__()
#         self.w = nn.Linear(160, 160)
#         self.q = nn.Linear(160, 1, bias=False)
#
#     def forward(self, features):
#         features = torch.stack(features, dim=1)
#         self.attention = torch.tanh(self.q(self.w(features)))
#
#         a = torch.softmax(self.attention, 0)
#         # a = torch.unsqueeze(a, dim=-1)
#         # print(a.shape)
#         # print(features.shape)
#         features = a * features
#         return features.sum(dim=1)


# alphe
# class ProteinFusionAttention(nn.Module):
#     def __init__(self):
#         super(ProteinFusionAttention, self).__init__()
#         self.attention = nn.Parameter(torch.ones(2) / 2)
#
#     def forward(self, features):
#         features = torch.stack(features, dim=1)
#         # t, _ = features.max(dim=-1)
#         a = torch.softmax(self.attention, 0)
#         a = torch.unsqueeze(a, dim=-1)
#         features = a * features
#         return features.sum(dim=1)

# SE
# class ProteinFusionAttention(nn.Module):
#     def __init__(self):
#         super(ProteinFusionAttention, self).__init__()
#         self.attention = nn.Sequential(
#             nn.Linear(2, 2),
#             nn.ReLU(),
#             nn.Linear(2, 2),
#             nn.Sigmoid()
#         )
#
#     def forward(self, features):
#         features = torch.stack(features, dim=1)
#         t, _ = features.max(dim=-1)
#         a = self.attention(t)
#         a = torch.unsqueeze(a, dim=-1)
#         features = a * features
#         return features.mean(dim=1)

class DRHGTModel(nn.Module):
    def __init__(self, G, protein_dim, drug_dim, disease_dim, out_feature_dim, head_num=1, LayerNumber=4, train_G=None):
        super(DRHGTModel, self).__init__()
        self.GD = ('Protein', 'interaction_DG', 'Disease')
        self.TD = ('Protein', 'interaction_DT', 'Drug')

        self.DG = ('Disease', 'interaction_GD', 'Protein')
        self.DT = ('Drug', 'interaction_TD', 'Protein')

        self.G = G
        self.train_G = train_G
        # 返回特征到蛋白质
        self.feature_dim = out_feature_dim
        self.GDmodels = nn.ModuleList()
        self.TDmodels = nn.ModuleList()
        self.DGmodels = nn.ModuleList()
        self.DTmodels = nn.ModuleList()
        self.Cmodels = nn.ModuleList()

        # protein_dim = 160
        self.layerNumber = LayerNumber
        self.headNumber = 1
        self.haveNorm = True
        self.proteinFusionAttention = ProteinFusionAttention()
        self.drugFusionAttention = FeatureFusionAttention(self.layerNumber)
        self.diseaseFusionAttention = FeatureFusionAttention(self.layerNumber)
        self.DrugMLP = nn.Sequential(nn.Linear(drug_dim, 160))
        self.DiseaseMLP = nn.Sequential(nn.Linear(disease_dim, 160))
        self.ProteinMLP = nn.Sequential(nn.Linear(protein_dim, 160))

        for i in range(self.layerNumber):
            if (i == 0):
                self.GDmodels.append(HGTLayer.HGTAttention(protein_dim, disease_dim, self.feature_dim, self.feature_dim,
                                                           self.GD, n_heads=self.headNumber,
                                                           use_norm=self.haveNorm))
                self.TDmodels.append(HGTLayer.HGTAttention(protein_dim, drug_dim, self.feature_dim, self.feature_dim,
                                                           self.TD, n_heads=self.headNumber,
                                                           use_norm=self.haveNorm))
                self.DGmodels.append(HGTLayer.HGTAttention(disease_dim, protein_dim, self.feature_dim,
                                                           self.feature_dim,
                                                           self.DG, n_heads=self.headNumber,
                                                           use_norm=self.haveNorm))

                self.DTmodels.append(HGTLayer.HGTAttention(drug_dim, protein_dim, self.feature_dim,
                                                           self.feature_dim,
                                                           self.DT, n_heads=self.headNumber,
                                                           use_norm=self.haveNorm))
                self.Cmodels.append(MLPPredictor(self.feature_dim * 2))
            else:
                self.GDmodels.append(
                    HGTLayer.HGTAttention(self.feature_dim, self.feature_dim, self.feature_dim, self.feature_dim,
                                          self.GD, n_heads=self.headNumber,
                                          use_norm=self.haveNorm))
                self.TDmodels.append(
                    HGTLayer.HGTAttention(self.feature_dim, self.feature_dim, self.feature_dim, self.feature_dim,
                                          self.TD, n_heads=self.headNumber,
                                          use_norm=self.haveNorm))
                self.DGmodels.append(HGTLayer.HGTAttention(self.feature_dim, self.feature_dim, self.feature_dim,
                                                           self.feature_dim,
                                                           self.DG, n_heads=self.headNumber,
                                                           use_norm=self.haveNorm))

                self.DTmodels.append(HGTLayer.HGTAttention(self.feature_dim, self.feature_dim, self.feature_dim,
                                                           self.feature_dim,
                                                           self.DT, n_heads=self.headNumber,
                                                           use_norm=self.haveNorm))
                self.Cmodels.append(MLPPredictor(self.feature_dim * 2))

        self.MLPLiner = MLPPredictor(self.feature_dim * LayerNumber * 2)

        self.attention = nn.Parameter(torch.ones(self.layerNumber) / self.layerNumber)

        init_weights(self)

    def noise(self, data):
        return data + torch.rand_like(data) * 0.005

    def forward(self, sub_edge, alhpe=1):
        with self.G.local_scope():
            self.a = torch.softmax(self.attention, 0)
            # drug_features = [self.DrugMLP(self.G.nodes['Drug'].data[f"h0"])]
            # disease_features = [self.DiseaseMLP(self.G.nodes['Disease'].data[f"h0"])]
            # protein_features = [self.ProteinMLP(self.G.nodes['Protein'].data[f"h0"])]
            drug_features = []
            disease_features = []
            for i in range(self.layerNumber):
                self.DTmodels[i](self.G, f"h{i}", f"h{i}", f"drug{i + 1}", alhpe)
                self.DGmodels[i](self.G, f"h{i}", f"h{i}", f"disease{i + 1}", alhpe)
                self.TDmodels[i](self.G, f"h{i}", f"h{i}", f"h{i + 1}", alhpe)
                self.GDmodels[i](self.G, f"h{i}", f"h{i}", f"h{i + 1}", alhpe)
                self.G.nodes['Protein'].data[f'h{i + 1}'] = self.proteinFusionAttention(
                    [self.G.nodes['Protein'].data[f"drug{i + 1}"],
                     self.G.nodes['Protein'].data[f"disease{i + 1}"]])
                # self.G.nodes['Protein'].data[f'h{i + 1}'] = self.G.nodes['Protein'].data[f'h{i + 1}'] + \
                #                                             protein_features[-1]
                # self.G.nodes['Drug'].data[f"h{i + 1}"] = self.G.nodes['Drug'].data[f"h{i + 1}"] + drug_features[-1]
                # self.G.nodes['Disease'].data[f"h{i + 1}"] = self.G.nodes['Disease'].data[f"h{i + 1}"] + \
                #                                             disease_features[-1]
                # drug_features.append(self.G.nodes['Drug'].data[f"h{i + 1}"] + drug_features[-1])
                # disease_features.append(self.G.nodes['Disease'].data[f"h{i + 1}"] + disease_features[-1])
                drug_features.append(self.G.nodes['Drug'].data[f"h{i + 1}"])
                disease_features.append(self.G.nodes['Disease'].data[f"h{i + 1}"])

                # if (i == 0):
                #     score = self.a[i] * self.Cmodels[i](sub_edge, self.G.nodes['Drug'].data[f"h{i + 1}"],
                #                                         self.G.nodes['Disease'].data[f"h{i + 1}"])
                # else:
                #     score += self.a[i] * self.Cmodels[i](sub_edge, self.G.nodes['Drug'].data[f"h{i + 1}"],
                #                                          self.G.nodes['Disease'].data[f"h{i + 1}"])
                # if (i == 0):
                #     score = 1 / self.layerNumber * self.Cmodels[i](sub_edge, self.G.nodes['Drug'].data[f"h{i + 1}"],
                #                                                    self.G.nodes['Disease'].data[f"h{i + 1}"])
                # else:
                #     score += 1 / self.layerNumber *self.Cmodels[i](sub_edge, self.G.nodes['Drug'].data[f"h{i + 1}"],
                #                                          self.G.nodes['Disease'].data[f"h{i + 1}"])
            # if (self.training == False):
            #     print(self.a)
            # Res
            # drug_feature = self.drugFusionAttention(drug_features)
            # disease_feature = self.diseaseFusionAttention(disease_features)
            # y = self.Cmodels[0](sub_edge, drug_feature, disease_feature)
            # 融合方式消融 - 多层注意力
            # drug_feature = self.drugFusionAttention(drug_features)
            # disease_feature = self.diseaseFusionAttention(disease_features)
            # y = self.Cmodels[0](sub_edge, drug_feature, disease_feature)
            # 融合方式消融 - Res+多层注意力
            # drug_feature = self.drugFusionAttention(drug_features[1:])
            # disease_feature = self.diseaseFusionAttention(disease_features[1:])
            # y = self.Cmodels[0](sub_edge, drug_feature, disease_feature)
            # 融合方式消融-多层特征融合
            drug_feature = self.drugFusionAttention(drug_features)
            disease_feature = self.diseaseFusionAttention(disease_features)
            y = self.MLPLiner(sub_edge, drug_feature, disease_feature)

            # 融合方式消融-ResAttention
            # drug_feature = self.drugFusionAttention(drug_features[1:])
            # disease_feature = self.diseaseFusionAttention(disease_features[1:])
            # y = self.Cmodels[4](sub_edge, drug_feature, disease_feature)
            # return y
            # end
            # 融合方式消融-只取最后一层
            # y = self.Cmodels[0](sub_edge, self.G.nodes['Drug'].data[f"h{self.layerNumber}"],
            #                     self.G.nodes['Disease'].data[f"h{self.layerNumber}"])
            # y = self.Cmodels[4](sub_edge, drug_feature, disease_feature)
            return y
            # end
            # return torch.clamp(score, 0, 1)
