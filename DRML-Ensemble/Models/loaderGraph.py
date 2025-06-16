import time
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from sklearn.metrics import roc_auc_score
from torch.utils import data
import ProteinsInteractions.Dataset as Dataset
from ProteinsInteractions.Models import DNN, AutoEncoder, GCN
from sklearn.decomposition import PCA
from sklearn import metrics
import os
import CommonHHJ.CsvTools as csv
import random
import scipy.io
import urllib.request
import dgl
import math
import numpy as np
import pandas as pd
import torch as th


def CreateTestData(N, M, path):
    map = np.random.randn(N, M)
    data = []
    for i in range(N):
        for j in range(M):
            if map[i][j] > 0.8:
                data.append([i, j, 0])
    csv.WriteData(path, data)


def CreateTestFeature(N, L, path):
    features = []
    for i in range(N):
        feature = np.zeros(N)
        feature[i] = 1
        features.append([i] + feature.tolist())
    csv.WriteData(path, features)
    # map = np.random.randn(N, L)
    # csv.WriteData(path, map.tolist())


def GetAllNode(datasetManager_train, datasetManager_test):
    train_data = datasetManager_train.getAllData().data
    test_data = datasetManager_test.getAllData().data
    graph = set()

    for x, y, l in train_data:
        graph.add(x)
        graph.add(y)

    for x, y, l in test_data:
        graph.add(x)
        graph.add(y)

    nodes = list(graph)

    feature_x = []
    for i in nodes:
        feature_x.append(Dataset.ProteinDatasetManager.getFeature(i))
    feature_x = torch.from_numpy(np.array(feature_x)).float()

    return nodes, feature_x


def load_edge_data(train_data, nodes):
    train_data = train_data.data.tolist()
    # train_data += val_data
    neg_edge_train_v = []
    neg_edge_train_u = []

    pos_edge_train_v = []
    pos_edge_train_u = []

    edge_train_v = []
    edge_train_u = []

    for x, y, l in train_data:
        if (l == 1):
            edge_train_v.append(nodes.index(x))
            edge_train_u.append(nodes.index(y))

            pos_edge_train_v.append(nodes.index(x))
            pos_edge_train_u.append(nodes.index(y))
        elif (l == 0):
            neg_edge_train_v.append(nodes.index(x))
            neg_edge_train_u.append(nodes.index(y))

    train_g = dgl.graph((edge_train_v, edge_train_u), num_nodes=len(nodes))

    train_pos_g = dgl.graph((pos_edge_train_v, pos_edge_train_u), num_nodes=len(nodes))
    train_neg_g = dgl.graph((neg_edge_train_v, neg_edge_train_u), num_nodes=len(nodes))

    return train_g, train_pos_g, train_neg_g


class GraphInfo():
    def __init__(self, path, srctype, tartype, etype):
        self.path = path
        self.srctype = srctype
        self.tartype = tartype
        self.etype = etype

    def getInfo(self):
        return self.path, self.srctype, self.tartype, self.etype

    def GraphInfo(self):
        return str.split(self.srctype, '_')[0], self.etype, str.split(self.tartype, '_')[0]


class HeterogeneousGraphManager():
    def __init__(self):
        self.nodesDic = {}
        self.g = None

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
        path, srctype, tartype, etype = graphInfo.getInfo()
        graph = pd.read_csv(path)

        srcNodeDic, srcIndexDic = self.loadNodes(graph, srctype)
        tarNodeDic, tarIndexDic = self.loadNodes(graph, tartype)
        v = []
        u = []
        graph = graph.values
        for i in graph:
            v.append(srcNodeDic[i[0]])
            u.append(tarNodeDic[i[1]])
        v = th.tensor(v)
        u = th.tensor(u)
        return (v, u)

    def loadHeterogeneousGraph(self, graphInfos):
        # graphs 是一个列表结构,每一个元素是 (path,srctype,tartype,etype)
        HeterogeneousGraph = {}
        for graphInfo in graphInfos:
            HeterogeneousGraph[graphInfo.GraphInfo()] = self.loadGraph(graphInfo)
        # print(HeterogeneousGraph)
        self.g = dgl.heterograph(HeterogeneousGraph)

    def loadFeature(self, path, featuresName, type):
        features = pd.read_csv(path, header=None)
        keys = features.values[:, 0]
        features = dict(zip(keys, features.values[:, 1:]))
        nodeIndexDic = self.nodesDic[type][1]
        nodeNums = len(nodeIndexDic)
        # print(nodeNums)
        data = []
        for i in range(nodeNums):
            key = nodeIndexDic[i]
            data.append(features[key])

        self.g.nodes[type].data[featuresName] = th.Tensor(data)
        # print(keys)
        # print(features)


if __name__ == '__main__':
    # 药物-靶标网络
    # CreateTestData(100, 100, "DTI.csv")
    # 疾病-基因网络
    # CreateTestData(100, 100, "DGI.csv")
    # 药物-药物网络
    # CreateTestData(100, 100, "DrugDI.csv")
    # 疾病-疾病网络
    # CreateTestData(100, 100, "DisDI.csv")
    # 疾病药物网络
    # CreateTestData(100, 100, "DrugDisI.csv")
    # 蛋白质特征
    # CreateTestFeature(100, 320, "TFeature.csv")

    m = HeterogeneousGraphManager()
    graphs = [
        GraphInfo("dataset/testdata/DGI.csv", "Disease", "Protein", "interaction"),
        GraphInfo("dataset/testdata/DisDI.csv", "Disease_A", "Disease_B", "interaction"),
        GraphInfo("dataset/testdata/DrugDI.csv", "Drug_A", "Drug_B", "interaction"),
        GraphInfo("dataset/testdata/DrugDisI.csv", "Drug", "Disease", "treath"),
        GraphInfo("dataset/testdata/DTI.csv", "Drug", "Protein", "interaction")
    ]
    m.loadHeterogeneousGraph(graphs)
    m.loadFeature("dataset/testdata/TFeature.csv", "x", "Protein")
    # print(m.g)
