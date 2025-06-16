import sys

sys.path.append('/sdb2/20721511/')
sys.path.append('/home/20721511/')
import pandas as pd
import src.dataloader as dataloader
import argparse
import torch.nn.functional as F
import numpy as np
import torch
import random
import CommonHHJ.CsvTools as csv
from src.metric_fn import evaluate
from torch.utils import data

torch.set_printoptions(profile="full")


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


def parse(print_help=True):
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasetRoot", default="CDatasetAll", type=str)
    parser.add_argument("--featureDim", default=320, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--epoch", default=500, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--feature_have", default=True, type=bool)
    parser.add_argument("--type", default="EMSFeature_HGT_2L320_5e-4Layer_norm_NewDiseasefeature_xavier",
                        type=str)
    parser.add_argument("--agg_type", default="mean", type=str)
    parser.add_argument("--feature_path", default="TFeature_sigmoid.csv", type=str)
    args = parser.parse_args()
    if print_help:
        parser.print_help()
    print(args)
    return args


if __name__ == '__main__':
    args = parse()
    setup_seed(args.seed)
    print(torch.__version__)
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
        dataloader.GraphInfo(f"{args.datasetRoot}/DisDI.csv", "Disease_A", "interaction_DisD", "Disease_B"),
        dataloader.GraphInfo(f"{args.datasetRoot}/DrugDI.csv", "Drug_A", "interaction_DrugD", "Drug_B"),
    ]

    heterogeneousGraphManager.loadHeterogeneousGraph(graphs)
    heterogeneousGraphManager.g.nodes['Protein'].data[f'h0'] = torch.zeros(5939, 160).to(device)
    drug_dim = heterogeneousGraphManager.loadFeature(f"{args.datasetRoot}/DrugFeature.csv", "h0", "Drug", isNorm=False,
                                                     topk=60)
    disease_dim = heterogeneousGraphManager.loadFeature(f"{args.datasetRoot}/NewDiseaseFeature.csv", "h0", "Disease",
                                                        isNorm=False, topk=60)

    print(heterogeneousGraphManager.g)
    etype = ("Drug", "treath", "Disease")
    protein_dim = 160
    heterogeneousGraphManager.constructNegativeGraphALL(etype)
    pos_row, pos_col = heterogeneousGraphManager.g.edges(etype=etype)
    repurposing_data = dataloader.DRDataset((pos_row, pos_col), heterogeneousGraphManager.neg_graph)
    batch_size = repurposing_data.__len__()
    repurposing_dataloader = data.DataLoader(repurposing_data, batch_size=batch_size, shuffle=True,
                                             num_workers=0,
                                             pin_memory=True,
                                             worker_init_fn=np.random.seed(0))
    root = "最终结果/topk消融/ZeroFeature160-6L_top60_alphe_0.2_newDiseaseFeature_topK消融6L"
    with torch.no_grad():
        for i in range(10):
            model = torch.load(f"{root}/Model/bestmodel_{i}.plt")
            model.eval()
            for data_row, data_cow, label in repurposing_dataloader:
                sub_train = heterogeneousGraphManager.createGraph(data_row, data_cow, label, etype)
                sub_train = sub_train.to(device)
                score = model(sub_train, 0.2)
                score = score.cpu().numpy().tolist()
                row = data_row.cpu().numpy().tolist()
                cow = data_cow.cpu().numpy().tolist()
                label = label.cpu().numpy().tolist()
                result = np.array([row, cow, score, label]).T
                result = result.tolist()
                csv.WriteData(f"{root}/repurposing/repurposing{i}.csv", result)
    summry = {}
    for c in range(10):
        score = pd.read_csv(f"{root}/repurposing/repurposing{c}.csv", header=None)
        score = score.values
        data = [["Drug", "Disease", "Score", "Label"]]
        for i in score:
            Drug_name = heterogeneousGraphManager.nodesDic['Drug'][1][i[0]]
            Disease_name = heterogeneousGraphManager.nodesDic['Disease'][1][i[1]]
            if (summry.keys().__contains__((Drug_name, Disease_name)) == False):
                summry[(Drug_name, Disease_name)] = [0, 0]
            summry[(Drug_name, Disease_name)][0] += i[2]
            summry[(Drug_name, Disease_name)][1] += i[3]

            data.append([Drug_name, Disease_name, i[2], i[3]])
        csv.WriteData(f"{root}/repurposing/repurposing_name{c}.csv", data)

    data = [["Drug", "Disease", "Score", "Label"]]
    for i in summry.keys():
        data.append([i[0], i[1], summry[i][0], summry[i][1]])
    csv.WriteData(f"{root}/repurposing/repurposing_Finally.csv", data)
    print(summry)
