import math
import sys

sys.path.append('/sdb2/20721511/')
sys.path.append('/home/20721511/')
import pandas as pd
import src.dataloader as dataloader
import argparse
import torch.nn.functional as F
import numpy as np
import dgl
import torch
import random
import Models.Model as models
import Models.FocalLoss as focaloss
from sklearn import metrics
import CommonHHJ.CsvTools as csv
from src.metric_fn import evaluate
from torch.utils import data
import Models.util as util
from dgl import DropEdge


def setup_seed(s):
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    np.random.seed(s)
    # torch.random.seed()
    random.seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_loss(scores, labels, device, loss_wight=None):
    # scores = torch.cat([pos_score, neg_score])
    # labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).long().to(device)
    # labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy(scores, labels.float())
    #
    # loss = focaloss.FocalLoss(alpha=0.25, gamma=2).to(device)
    # scores = torch.softmax(scores,dim=1)
    # return loss(scores, labels)

    # return F.cross_entropy(scores, labels, loss_wight)


def compute_auc(scores, labels, i, isSave):
    if (isSave):
        d = np.concatenate((torch.unsqueeze(scores, dim=1).cpu().numpy(), torch.unsqueeze(labels, dim=1).cpu().numpy()),
                           axis=1)
        csv.WriteData("data.csv", d.tolist())
    # pre_value_Other = torch.cat([pos_score[:, 1], neg_score[:, 1]]).cpu().numpy()
    # true_label = torch.cat(
    #     [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    # scores = scores[:, 1].cpu().numpy()
    print(scores[0], labels[0])
    scores = scores.cpu().numpy()
    labels = labels.cpu().numpy()
    # print(scores.shape)
    # print(labels.shape)
    r = evaluate(scores, labels)
    print(r)

    return r


def parse(print_help=True, topK=30, alphe=1.0, LayerNumber=4, T=""):
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasetRoot", default="CAllGraph", type=str)
    parser.add_argument("--featureDim", default=160, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--LayerNumber", default=LayerNumber, type=int)
    parser.add_argument("--epoch", default=300, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--alphe", default=alphe, type=float)
    parser.add_argument("--topK", default=topK, type=int)
    parser.add_argument("--feature_have", default=True, type=bool)
    # parser.add_argument("--type", default="RFeature_MyHGT_640_notBN_bs2000_all_focalLossAlpha_0.25_gamma2_e100", type=str)
    # parser.add_argument("--type",
    #                     default=f"DNN",
    #                     type=str)
    parser.add_argument("--type",
                        default=f"ZeroFeature160-{LayerNumber}L_top{topK}_alphe_{alphe}_newDiseaseFeature_{T}",
                        type=str)
    # parser.add_argument("--type", default="DNNencoder160Feature_Sample_sum", type=str)
    parser.add_argument("--agg_type", default="mean", type=str)
    # parser.add_argument("--feature_path", default="TFeature_R160.csv", type=str)
    # parser.add_argument("--feature_path", default="TFeature_sigmoid.csv", type=str)
    parser.add_argument("--feature_path", default="TFeature_Z160.csv", type=str)
    # parser.add_argument("--feature_path", default="TFeature1280.csv", type=str)

    args = parser.parse_args()
    if print_help:
        parser.print_help()
    print(args)
    return args


def train(args):
    setup_seed(args.seed)
    # 数据读取
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    print(device)
    heterogeneousGraphManager = dataloader.HeterogeneousGraphManager(args, device)
    graphs = [
        dataloader.GraphInfo(f"{args.datasetRoot}/DTI.csv", "Protein", "interaction_DT", "Drug"),
        dataloader.GraphInfo(f"{args.datasetRoot}/DGI.csv", "Protein", "interaction_DG", "Disease"),
        dataloader.GraphInfo(f"{args.datasetRoot}/DrugDisI.csv", "Drug", "treath", "Disease"),
        dataloader.GraphInfo(f"{args.datasetRoot}/DTI.csv", "Drug", "interaction_TD", "Protein"),
        dataloader.GraphInfo(f"{args.datasetRoot}/DGI.csv", "Disease", "interaction_GD", "Protein"),
    ]
    heterogeneousGraphManager.loadHeterogeneousGraph(graphs)
    # protein_dim = heterogeneousGraphManager.loadFeature(f"{args.datasetRoot}/{args.feature_path}", "h0", "Protein",
    #                                                     isNorm=False)
    # heterogeneousGraphManager.g.nodes['Protein'].data[f'h0'] = torch.zeros(5846, 160).to(device)
    heterogeneousGraphManager.g.nodes['Protein'].data[f'h0'] = torch.zeros(5939, 160).to(device)

    # heterogeneousGraphManager.g.nodes['Protein'].data[f'h0'] = torch.zeros(4880, 160).to(device)
    # heterogeneousGraphManager.g.nodes['Protein'].data[f'h0'] = torch.zeros(4808, 160).to(device)
    protein_dim = 160
    topK = args.topK
    LayerNumber = args.LayerNumber

    drug_dim = heterogeneousGraphManager.loadFeature(f"{args.datasetRoot}/DrugFeature.csv", "h0", "Drug", isNorm=False,
                                                     topk=topK)
    disease_dim = heterogeneousGraphManager.loadFeature(f"{args.datasetRoot}/DiseaseFeature.csv", "h0", "Disease",
                                                        isNorm=False, topk=topK)

    print(heterogeneousGraphManager.g)
    etype = ("Drug", "treath", "Disease")
    neg_mask = []
    type_lr = args.type
    epoch = args.epoch
    lr = args.lr
    result = []
    result_mean = []

    # 特征检验
    # model = models.SampleSumModel(heterogeneousGraphManager.g, args.featureDim, args.agg_type, args.feature_have)
    # print(heterogeneousGraphManager.g.nodes["Disease"].data["he"])
    # t = heterogeneousGraphManager.g.nodes["Drug"].data["he"]
    # # print(torch.sum(heterogeneousGraphManager.g.nodes["Disease"].data["he"][1]))
    # for i in t:
    #     print(torch.sum(i))
    # 模型训练

    # heterogeneousGraphManager.graphKFoldDrug(10, etype)
    # 特征检验
    # model = models.SampleSumModel(heterogeneousGraphManager.g, args.featureDim, args.agg_type, args.feature_have)
    # print(heterogeneousGraphManager.g.nodes["Disease"].data["he"])
    # t = heterogeneousGraphManager.g.nodes["Drug"].data["he"]
    # # print(torch.sum(heterogeneousGraphManager.g.nodes["Disease"].data["he"][1]))
    # for i in t:
    #     print(torch.sum(i))
    # 模型训练
    for i, (wight, train_pos, train_neg, test_pos, test_neg) in enumerate(
            heterogeneousGraphManager.graphKFold(10, etype, 1, return_graph=False)):
        # if (i != 1):
        #     continue
        # 数据集划分
        train_dataset = dataloader.DRDataset(train_pos, train_neg)
        batch_size = train_dataset.__len__()
        train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                           num_workers=0,
                                           pin_memory=True,
                                           worker_init_fn=np.random.seed(0))
        test_dataset = dataloader.DRDataset(test_pos, test_neg)
        test_dataloader = data.DataLoader(test_dataset, batch_size=test_dataset.__len__(), shuffle=True,
                                          num_workers=0,
                                          pin_memory=True,
                                          worker_init_fn=np.random.seed(0))

        print(f"Train: {i}")
        # 创建模型
        # model = models.SampleSumModel(heterogeneousGraphManager.g, args.featureDim, args.agg_type, args.feature_have)
        # model = models.SimModel(heterogeneousGraphManager.g, args.featureDim, args.feature_have)
        model = models.DRHGTModel(heterogeneousGraphManager.g, protein_dim, drug_dim, disease_dim, args.featureDim,
                                  LayerNumber=LayerNumber)

        model = model.to(torch.device(device))

        # 创建优化器设置
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
        loss_wight = torch.tensor([wight]).to(device)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.90)

        # 实验参数设置记录
        csv.WriteData(f"{args.datasetRoot}/{type_lr}/Model/data.csv", [args, "在特征转换器后加入tanh"])
        result_loss = []
        alphe = args.alphe
        # 训练
        for e in range(epoch):
            loss_train = 0
            c = 0
            # dropEdge = util.MyDropEdge(0.5)
            # heterogeneousGraphManager.g = dropEdge(heterogeneousGraphManager.g,
            #                                        [('Protein', 'interaction_DG', 'Disease'),
            #                                         ("Disease", "interaction_GD", "Protein"),
            #                                         ("Drug", "interaction_TD", "Protein"),
            #                                         ("Protein", "interaction_DT", "Drug")])
            # print(Hg.edges[('Protein', 'interaction_DG', 'Disease')].data)

            for data_row, data_cow, label in train_dataloader:
                sub_train = heterogeneousGraphManager.createGraph(data_row, data_cow, label, etype)
                sub_train = sub_train.to(device)
                model.train()
                scores = model(sub_train, alphe)
                loss = compute_loss(scores, sub_train.edges[etype[1]].data['score'], device, loss_wight)
                loss_train += loss.item()
                c += 1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
            # torch.save(model, f"{args.datasetRoot}/{type_lr}/Model/bestmodel_{i}.plt")

            # 验证
            with torch.no_grad():
                if (e % 1 == 0):
                    model.eval()
                    # print("Train Result")
                    for data_row, data_cow, label in test_dataloader:
                        sub_test = heterogeneousGraphManager.createGraph(data_row, data_cow, label, etype)
                        sub_test = sub_test.to(device)
                        test_scores = model(sub_test, alphe)
                        loss_val = compute_loss(test_scores, sub_test.edges[etype[1]].data['score'], device, loss_wight)
                        compute_auc(test_scores, sub_test.edges[etype[1]].data['score'], i, False)
                        print(f"{e}====train_loss:{loss_train / c},val_loss:{loss_val.item()}")
                        result_loss.append([e, loss_train / c, loss_val.item()])
                        torch.save(model, f"{args.datasetRoot}/{type_lr}/Model/bestmodel_{i}.plt")

        # 测试
        best_model = torch.load(f"{args.datasetRoot}/{type_lr}/Model/bestmodel_{i}.plt")
        best_model.eval()
        with torch.no_grad():
            print("test,Result")
            for data_row, data_cow, label in test_dataloader:
                sub_test = heterogeneousGraphManager.createGraph(data_row, data_cow, label, etype)
                sub_test = sub_test.to(device)
                test_scores = best_model(sub_test, alphe)
                r = compute_auc(test_scores, sub_test.edges[etype[1]].data['score'], i, True)
                result.append(
                    [r['aupr'], r['auroc'], r['lagcn_aupr'], r['lagcn_auc'], r['lagcn_f1_score'], r['lagcn_accuracy'],
                     r['lagcn_recall'], r['lagcn_specificity'], r['lagcn_precision']])

        np_r = np.array(result)
        print(np.mean(np_r, axis=0))
        result_mean.append(np.mean(np_r, axis=0).tolist())
        csv.WriteData(f"{args.datasetRoot}/{type_lr}/Result.csv", result)
        csv.WriteData(f"{args.datasetRoot}/{type_lr}/Result_loss_{i}.csv", result_loss)
        csv.WriteData(f"{args.datasetRoot}/{type_lr}/Result_mean.csv", result_mean)


if __name__ == '__main__':
    # args = parse(topK=30, alphe=0.1, T="十字节")
    # train(args)
    # args = parse(topK=30, alphe=0.2,T="2")
    # train(args)
    # args = parse(topK=i, alphe=0.2, LayerNumber=5, T="topK消融5L")

    # alphes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    # for i in alphes:
    #     args = parse(topK=15, alphe=0.1, LayerNumber=i, T="原始药物特征层数消融")
    #     train(args)

    # alphes = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    # for i in alphes:
    #     args = parse(topK=15, a lphe=i, LayerNumber=8, T="原始药物特征alphe消融3")
    #     train(args)

    # tops = [1, 15, 30, 45, 60, 75, 90, 105, 120, -1]
    # for i in tops:
    #     args = parse(topK=i, alphe=0.1, LayerNumber=8, T="原始药物特征top消融")
    #     train(args)

    # layers = [9, 10, 11, 12]
    # layers = [2]
    # for i in layers:
    #     args = parse(topK=60, alphe=0.2, LayerNumber=i, T="没有alphe")
    #     train(args)

    # args = parse(topK=15, alphe=0.1, LayerNumber=8, T="多特征拼接")
    # args = parse(topK=60, alphe=0.2, LayerNumber=6, T="多层特征融合")
    # args = parse(topK=60, alphe=0.2, LayerNumber=6, T="原始验证1")

    # args = parse(topK=15, alphe=0.1, LayerNumber=8, T="原始药物特征alphe消融3_BNnotInit_DRTerHGAT")
    # train(args)
    args = parse(topK=60, alphe=0.2, LayerNumber=6, T="")
    train(args)
