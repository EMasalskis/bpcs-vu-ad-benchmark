import scipy.io
import urllib.request
import dgl
import math
import numpy as np
import pandas as pd
import torch
import Models.Model as models
from sklearn.model_selection import KFold
import random
from torch.utils import data


class GraphInfo():
    def __init__(self, path, srctype, etype, tartype, weight=None):
        self.path = path
        self.srctype = srctype
        self.tartype = tartype
        self.etype = etype
        self.weight = weight

    def getInfo(self):
        return self.path, self.srctype, self.tartype, self.etype, self.weight

    def GraphInfo(self):
        return (str.split(self.srctype, '_')[0], self.etype, str.split(self.tartype, '_')[0]), self.weight


class HeterogeneousGraphManager():
    def __init__(self, args, device):
        self.nodesDic = {}
        self.args = args
        self.g = None
        self.neg_graph = None
        self.device = device

    def loadNodes(self, graph_data, type):
        data = graph_data[type]
        data = data.values
        nodeType = str.split(type, '_')[0]
        index = 0

        if (self.nodesDic.keys().__contains__(nodeType) == False):
            self.nodesDic[nodeType] = ({}, {})
        else:
            index = len(self.nodesDic[nodeType][0].keys())

        nodeDic = self.nodesDic[nodeType][0]
        indexDic = self.nodesDic[nodeType][1]
        for i in data:
            if (nodeDic.keys().__contains__(i) == False):
                nodeDic[i] = index
                indexDic[index] = i
                index = index + 1

        self.nodesDic[nodeType] = (nodeDic, indexDic)
        # 返回双向字典 第一个 节点名称：序号， 第二个 序号：节点名称
        return (nodeDic, indexDic)

    def loadGraph(self, graphInfo):
        path, srctype, tartype, etype, weight = graphInfo.getInfo()
        graph = pd.read_csv(path)

        srcNodeDic, srcIndexDic = self.loadNodes(graph, srctype)
        tarNodeDic, tarIndexDic = self.loadNodes(graph, tartype)
        v = []
        u = []

        graph_edge = graph[[srctype, tartype]].values
        for i in graph_edge:
            v.append(srcNodeDic[i[0]])
            u.append(tarNodeDic[i[1]])
        v = torch.tensor(v)
        u = torch.tensor(u)
        w = None
        if (weight != None):
            wdata = graph[weight].values
            w = torch.from_numpy(wdata).float()
        return (v, u), w

    def loadHeterogeneousGraph(self, graphInfos):
        # graphs 是一个列表结构,每一个元素是 (path,srctype,tartype,etype)
        HeterogeneousGraph = {}
        weightGraph = {}
        for graphInfo in graphInfos:
            graph, weight = self.loadGraph(graphInfo)
            graph_name, weight_name = graphInfo.GraphInfo()
            HeterogeneousGraph[graph_name] = graph
            if (weight != None):
                weightGraph[graph_name] = (weight_name, weight)

        self.g = dgl.heterograph(HeterogeneousGraph)
        for w in weightGraph:
            self.g.edges[w[1]].data[weightGraph[w][0]] = weightGraph[w][1]
        self.g = self.g.to(self.device)

    def loadFeature(self, path, featuresName, type, isNorm=True, topk=-1):
        features = pd.read_csv(path, header=None)

        keys = features.values[:, 0]
        feature = features.values[:, 1:]
        if (isNorm):
            f = feature.astype(np.float32)
            std = np.std(f, axis=0, keepdims=True)
            mean = np.mean(f, axis=0, keepdims=True)
            f = (f - mean) / (std + 1e-6)
            feature = f

        if (topk > 0):
            neighbor = np.argpartition(-feature, kth=topk, axis=1)[:, :topk]
            row_index = np.arange(neighbor.shape[0]).repeat(neighbor.shape[1])
            col_index = neighbor.reshape(-1)
            zero_m = np.zeros_like(feature)
            zero_m[row_index, col_index] = 1
            feature = feature * zero_m
            print(feature)

        features = dict(zip(keys, feature))
        nodeIndexDic = self.nodesDic[type][1]
        nodeNums = len(nodeIndexDic)
        data = []
        for i in range(nodeNums):
            key = nodeIndexDic[i]
            data.append(features[key])
        self.g.nodes[type].data[featuresName] = torch.Tensor(data).to(self.device)
        # print(self.g.nodes[type].data[featuresName][-1])
        # print(torch.Tensor(data).shape)
        # print(data[-1])
        return self.g.nodes[type].data[featuresName].shape[1]

    def repurposingDrugConstruct(self):
        etype = ("Drug", "treath", "Disease")
        utype, _, vtype = etype
        src, dst = self.g.edges(etype=etype)
        src = src.to("cpu").numpy()
        dst = dst.to("cpu").numpy()

        u = self.g.num_nodes(utype)
        v = self.g.num_nodes(vtype)
        pos_matrix = np.zeros((u, v))
        pos_matrix[src, dst] = 1
        neg_row, neg_col = np.nonzero(1 - pos_matrix)
        return

    def constructNegativeGraphALL(self, etype):
        utype, _, vtype = etype
        src, dst = self.g.edges(etype=etype)
        u = self.g.num_nodes(utype)
        v = self.g.num_nodes(vtype)
        s = set()
        for i in range(len(src)):
            s.add((src[i].item(), dst[i].item()))

        neg_src = []
        neg_dst = []
        for i in range(u):
            for j in range(v):
                if ((i, j) not in s):
                    neg_src.append(i)
                    neg_dst.append(j)

        neg_src = torch.LongTensor(neg_src)
        neg_dst = torch.LongTensor(neg_dst)
        self.neg_graph = (neg_src, neg_dst)

    def constructNegativeGraph(self, k, etype):
        utype, _, vtype = etype
        src, dst = self.g.edges(etype=etype)
        src = src.to("cpu").numpy()
        dst = dst.to("cpu").numpy()

        u = self.g.num_nodes(utype)
        v = self.g.num_nodes(vtype)
        pos_matrix = np.zeros((u, v))
        pos_matrix[src, dst] = 1
        neg_row, neg_col = np.nonzero(1 - pos_matrix)
        neg_num = neg_row.shape[0]
        src_num = src.shape[0]
        neg_index = np.random.choice(range(neg_num), size=k * src_num, replace=False)
        neg_src = torch.LongTensor(neg_row[neg_index])
        neg_dst = torch.LongTensor(neg_col[neg_index])
        self.neg_graph = (neg_src, neg_dst)

    def getAdjacent(self, edge):
        eg = dgl.edge_type_subgraph(self.g, [edge])
        adj = eg.adjacency_matrix()
        print(torch.zeros(adj.shape[0], adj.shape[1]))
        return adj

    def graphKFold(self, n_splits, etype, k, return_graph=True):
        self.constructNegativeGraph(k, etype)
        assert self.neg_graph != None
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=self.args.seed)
        pos_row, pos_col = self.g.edges(etype=etype)
        neg_row, neg_col = self.neg_graph
        for (train_pos_idx, test_pos_idx), (train_neg_idx, test_neg_idx) in zip(kfold.split(pos_row),
                                                                                kfold.split(neg_row)):
            if (return_graph):
                train_pos = dgl.heterograph({etype: (pos_row[train_pos_idx], pos_col[train_pos_idx])},
                                            num_nodes_dict={ntype: self.g.num_nodes(ntype) for ntype in self.g.ntypes})
                train_neg = dgl.heterograph({etype: (neg_row[train_neg_idx], neg_col[train_neg_idx])},
                                            num_nodes_dict={ntype: self.g.num_nodes(ntype) for ntype in self.g.ntypes})

                test_pos = dgl.heterograph({etype: (pos_row[test_pos_idx], pos_col[test_pos_idx])},
                                           num_nodes_dict={ntype: self.g.num_nodes(ntype) for ntype in self.g.ntypes})
                test_neg = dgl.heterograph({etype: (neg_row[test_neg_idx], neg_col[test_neg_idx])},
                                           num_nodes_dict={ntype: self.g.num_nodes(ntype) for ntype in self.g.ntypes})
                yield len(train_neg_idx) / len(train_pos_idx), train_pos, train_neg, test_pos, test_neg
            else:
                train_pos = (pos_row[train_pos_idx], pos_col[train_pos_idx])
                train_neg = (neg_row[train_neg_idx], neg_col[train_neg_idx])
                test_pos = (pos_row[test_pos_idx], pos_col[test_pos_idx])
                test_neg = (neg_row[test_neg_idx], neg_col[test_neg_idx])
                yield len(train_neg_idx) / len(train_pos_idx), train_pos, train_neg, test_pos, test_neg

    def getVal(self, index, p):
        l = len(index)
        train = int(p * l)
        train_index = np.random.choice(range(l), train, replace=False)
        val_index = []
        for i in range(l):
            if (i not in train_index):
                val_index.append(i)
        val_index = np.array(val_index)
        return index[train_index], index[val_index]

    def graphKFold_val(self, n_splits, etype, p=0.8):
        assert self.neg_graph != None
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=self.args.seed)
        pos_row, pos_col = self.g.edges(etype=etype)
        neg_row, neg_col = self.neg_graph
        s = set()
        for (train_pos_idx, test_pos_idx), (train_neg_idx, test_neg_idx) in zip(kfold.split(pos_row),
                                                                                kfold.split(neg_row)):
            train_pos_idx, val_pos_idx = self.Get_val(train_pos_idx, 0.8)
            train_neg_idx, val_neg_idx = self.Get_val(train_neg_idx, 0.8)

            train_pos = dgl.heterograph({etype: (pos_row[train_pos_idx], pos_col[train_pos_idx])},
                                        num_nodes_dict={ntype: self.g.num_nodes(ntype) for ntype in self.g.ntypes})
            train_neg = dgl.heterograph({etype: (neg_row[train_neg_idx], neg_col[train_neg_idx])},
                                        num_nodes_dict={ntype: self.g.num_nodes(ntype) for ntype in self.g.ntypes})

            val_pos = dgl.heterograph({etype: (pos_row[val_pos_idx], pos_col[val_pos_idx])},
                                      num_nodes_dict={ntype: self.g.num_nodes(ntype) for ntype in self.g.ntypes})
            val_neg = dgl.heterograph({etype: (neg_row[val_neg_idx], neg_col[val_neg_idx])},
                                      num_nodes_dict={ntype: self.g.num_nodes(ntype) for ntype in self.g.ntypes})

            test_pos = dgl.heterograph({etype: (pos_row[test_pos_idx], pos_col[test_pos_idx])},
                                       num_nodes_dict={ntype: self.g.num_nodes(ntype) for ntype in self.g.ntypes})
            test_neg = dgl.heterograph({etype: (neg_row[test_neg_idx], neg_col[test_neg_idx])},
                                       num_nodes_dict={ntype: self.g.num_nodes(ntype) for ntype in self.g.ntypes})

            yield len(train_neg_idx) / len(train_pos_idx), train_pos, train_neg, val_pos, val_neg, test_pos, test_neg

    def createGraph(self, data_row, data_cow, label, etype):
        sub_graph = dgl.heterograph({etype: (data_row, data_cow)},
                                    num_nodes_dict={ntype: self.g.num_nodes(ntype) for ntype in self.g.ntypes})
        sub_graph.edges[etype[1]].data['score'] = label
        return sub_graph


class DRDataset(data.Dataset):
    def __init__(self, pos_index, neg_index):
        self.pos_index = (pos_index[0].cpu(), pos_index[1].cpu())
        self.neg_index = (neg_index[0].cpu(), neg_index[1].cpu())
        self.pos_num = self.pos_index[0].shape[0]
        self.neg_num = self.neg_index[0].shape[0]
        self.Len = self.pos_num + self.neg_num
        self.data = (
            torch.cat([self.pos_index[0], self.neg_index[0]]), torch.cat([self.pos_index[1], self.neg_index[1]]))
        self.lable = torch.cat([torch.ones(self.pos_num), torch.zeros(self.neg_num)]).long()

    def __len__(self):
        return self.Len

    def __getitem__(self, index):
        return self.data[0][index], self.data[1][index], self.lable[index]


if __name__ == '__main__':
    m = HeterogeneousGraphManager(None)
    graphs = [
        GraphInfo("../dataset/testdata/DGI.csv", "Disease", "interaction_DG", "Protein"),
        GraphInfo("../dataset/testdata/DisDI.csv", "Disease_A", "interaction_DisD", "Disease_B"),
        GraphInfo("../dataset/testdata/DrugDI.csv", "Drug_A", "interaction_DrugD", "Drug_B"),
        GraphInfo("../dataset/testdata/DrugDisI.csv", "Drug", "treath", "Disease"),
        GraphInfo("../dataset/testdata/DTI.csv", "Drug", "interaction_DT", "Protein")
    ]
    m.loadHeterogeneousGraph(graphs)
    print(m.g.device)
    m.loadFeature("../dataset/testdata/TFeature.csv", "h", "Protein")
    # print(m.g)
    # m.g.node_dict = {}
    # m.g.edge_dict = {}
    # for ntype in m.g.ntypes:
    #     m.g.node_dict[ntype] = len(m.g.node_dict)
    #
    # for etype in m.g.etypes:
    #     # print(etype)
    #     m.g.edge_dict[etype] = len(m.g.edge_dict)
    #     m.g.edges[etype].data['id'] = torch.ones(m.g.number_of_edges(etype), dtype=torch.long) * m.g.edge_dict[etype]
    #
    # print(m.g.node_dict)
    model = models.SampleSumModel(m.g, 200)
    model = model.to("cuda:0")
    model(m.g)
    print(torch.sum(m.g.nodes['Drug'].data['h']))
    print(torch.sum(m.g.nodes['Disease'].data['h']))
