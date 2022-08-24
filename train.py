import networkx as nx
import numpy as np
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold
import sys
import scipy
import sklearn
import json
from collections import defaultdict
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import argparse
import math
import pickle as pkl
import scipy.sparse as sp
import time
import scipy.io as sio
import random
from sklearn import preprocessing
from sklearn.metrics import f1_score
# import contrast_util
import json
import os
from base_model_SSL import GCN_dense
from base_model_SSL import Linear
from base_model_SSL import GCN_emb


# from base_model import GCN

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def f1(output, labels):
    preds = output.max(1)[1].type_as(labels)
    f1 = f1_score(labels, preds, average='weighted')
    return f1


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def cal_euclidean(input):
    # input tensor
    #a = input.unsqueeze(0).repeat([input.shape[0], 1, 1])
    #b = input.unsqueeze(1).repeat([1, input.shape[0], 1])
    #distance = (a - b).square().sum(-1)
    distance = torch.cdist(input.unsqueeze(0),input.unsqueeze(0)).squeeze()
    
    return distance


def load_npz_to_sparse_graph(file_name):
    """Load a SparseGraph from a Numpy binary file.
    Parameters
    ----------
    file_name : str
        Name of the file to load.
    Returns
    -------
    sparse_graph : SparseGraph
        Graph in sparse matrix format.
    """
    with np.load(file_name) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
                                   shape=loader['adj_shape'])

        if 'attr_data' in loader:
            # Attributes are stored as a sparse CSR matrix
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'], loader['attr_indptr']),
                                        shape=loader['attr_shape'])
        elif 'attr_matrix' in loader:
            # Attributes are stored as a (dense) np.ndarray
            attr_matrix = loader['attr_matrix']
        else:
            attr_matrix = None

        if 'labels_data' in loader:
            # Labels are stored as a CSR matrix
            labels = sp.csr_matrix((loader['labels_data'], loader['labels_indices'], loader['labels_indptr']),
                                   shape=loader['labels_shape'])
        elif 'labels' in loader:
            # Labels are stored as a numpy array
            labels = loader['labels']
        else:
            labels = None

        node_names = loader.get('node_names')
        attr_names = loader.get('attr_names')
        class_names = loader.get('class_names')
        metadata = loader.get('metadata')

    return adj_matrix, attr_matrix, labels, node_names, attr_names, class_names, metadata


valid_num_dic = {'Amazon_clothing': 17, 'Amazon_eletronics': 36, 'dblp': 27}


def load_data_pretrain(dataset_source):
    if dataset_source == 'ogbn-arxiv':

        from ogb.nodeproppred import NodePropPredDataset
        dataset = NodePropPredDataset(name=dataset_source)


        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, labels = dataset[0]  # graph: library-agnostic graph object

        n1s = graph['edge_index'][0]
        n2s = graph['edge_index'][1]

        num_nodes = graph['num_nodes']
        print('nodes num', num_nodes)
        adj = sp.coo_matrix((np.ones(len(n1s)), (n1s, n2s)),
                            shape=(num_nodes, num_nodes))

        adj = normalize(adj + sp.eye(adj.shape[0]))
        adj = sparse_mx_to_torch_sparse_tensor(adj)

        features = torch.FloatTensor(graph['node_feat'])
        labels = torch.LongTensor(labels).squeeze()

        # class_list_test = random.sample(list(range(40)),20)
        # train_class=list(set(list(range(40))).difference(set(class_list_test)))
        # class_list_valid = random.sample(train_class, 5)
        # class_list_train = list(set(train_class).difference(set(class_list_valid)))
        # json.dump([class_list_train,class_list_valid,class_list_test],open('./few_shot_data/{}_class_split.json'.format(dataset_source),'w'))
        class_list_train, class_list_valid, class_list_test = json.load(
            open('./few_shot_data/{}_class_split.json'.format(dataset_source)))

        idx_train, idx_valid, idx_test = [], [], []

        for i in range(labels.shape[0]):
            if labels[i] in class_list_train:
                idx_train.append(i)
            elif labels[i] in class_list_valid:
                idx_valid.append(i)
            else:
                idx_test.append(i)
        print(labels.shape)

    elif dataset_source in valid_num_dic.keys():
        n1s = []
        n2s = []
        for line in open("./few_shot_data/{}_network".format(dataset_source)):
            n1, n2 = line.strip().split('\t')
            n1s.append(int(n1))
            n2s.append(int(n2))

        data_train = sio.loadmat("./few_shot_data/{}_train.mat".format(dataset_source))
        data_test = sio.loadmat("./few_shot_data/{}_test.mat".format(dataset_source))

        num_nodes = max(max(n1s), max(n2s)) + 1
        labels = np.zeros((num_nodes, 1))
        labels[data_train['Index']] = data_train["Label"]
        labels[data_test['Index']] = data_test["Label"]

        features = np.zeros((num_nodes, data_train["Attributes"].shape[1]))
        features[data_train['Index']] = data_train["Attributes"].toarray()
        features[data_test['Index']] = data_test["Attributes"].toarray()

        print('nodes num', num_nodes)
        adj = sp.coo_matrix((np.ones(len(n1s)), (n1s, n2s)),
                            shape=(num_nodes, num_nodes))

        class_list = []
        for cla in labels:
            if cla[0] not in class_list:
                class_list.append(cla[0])  # unsorted

        id_by_class = {}
        for i in class_list:
            id_by_class[i] = []
        for id, cla in enumerate(labels):
            id_by_class[cla[0]].append(id)

        lb = preprocessing.LabelBinarizer()
        labels = lb.fit_transform(labels)

        adj = normalize(adj + sp.eye(adj.shape[0]))
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(np.where(labels)[1])

        adj = sparse_mx_to_torch_sparse_tensor(adj)

        train_class = list(set(data_train["Label"].reshape((1, len(data_train["Label"])))[0]))
        class_list_test = list(set(data_test["Label"].reshape((1, len(data_test["Label"])))[0]))

        class_list_valid = random.sample(train_class, valid_num_dic[dataset_source])

        class_list_train = list(set(train_class).difference(set(class_list_valid)))

        # json.dump([class_list_train,class_list_valid,class_list_test],open('./few_shot_data/{}_class_split.json'.format(dataset_source),'w'))

        class_list_train, class_list_valid, class_list_test = json.load(
            open('./few_shot_data/{}_class_split.json'.format(dataset_source)))

        idx_train, idx_valid, idx_test = [], [], []
        for idx_, class_list_ in zip([idx_train, idx_valid, idx_test],
                                     [class_list_train, class_list_valid, class_list_test]):
            for class_ in class_list_:
                idx_.extend(id_by_class[class_])


    elif dataset_source == 'cora-full':
        adj, features, labels, node_names, attr_names, class_names, metadata = load_npz_to_sparse_graph(
            './dataset/gnn-benchmark/data/npz/cora_full.npz')

        sparse_mx = adj.tocoo().astype(np.float32)
        indices = np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)

        n1s = indices[0].tolist()
        n2s = indices[1].tolist()

        adj = normalize(adj.tocoo() + sp.eye(adj.shape[0]))
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        features = features.todense()
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(labels).squeeze()

        # features=features.todense()

        # class_list_test = random.sample(list(range(70)),25)
        # train_class=list(set(list(range(70))).difference(set(class_list_test)))
        # class_list_valid = random.sample(train_class, 20)
        # class_list_train = list(set(train_class).difference(set(class_list_valid)))
        # json.dump([class_list_train,class_list_valid,class_list_test],open('./few_shot_data/{}_class_split.json'.format(dataset_source),'w'))

        class_list_train, class_list_valid, class_list_test = json.load(
            open('./few_shot_data/{}_class_split.json'.format(dataset_source)))

        idx_train, idx_valid, idx_test = [], [], []

        for i in range(labels.shape[0]):
            if labels[i] in class_list_train:
                idx_train.append(i)
            elif labels[i] in class_list_valid:
                idx_valid.append(i)
            else:
                idx_test.append(i)

    class_train_dict = defaultdict(list)
    for one in class_list_train:
        for i, label in enumerate(labels.numpy().tolist()):
            if label == one:
                class_train_dict[one].append(i)

    class_test_dict = defaultdict(list)
    for one in class_list_test:
        for i, label in enumerate(labels.numpy().tolist()):
            if label == one:
                class_test_dict[one].append(i)

    print(len(idx_train))
    print(len(idx_train) + len(idx_valid))
    print(features.shape[0])

    
    write_to_g_meta=True
    if write_to_g_meta:
        import pandas as pd
        import dgl
        label_pkl={}
        name=[]
        label=[]
        for idx in idx_train:
            name.append('0_{}'.format(idx))
            label.append(labels[idx].item())
            label_pkl['0_{}'.format(idx)]= labels[idx].item()
            
        dataframe = pd.DataFrame({'a_name':name,'b_name':label})
        dataframe.to_csv("./G-Meta/new_data/{}/train.csv".format(dataset_source),sep=',')
        
        name=[]
        label=[]
        for idx in idx_valid:
            name.append('0_{}'.format(idx))
            label.append(labels[idx].item())
            label_pkl['0_{}'.format(idx)]= labels[idx].item()
        dataframe = pd.DataFrame({'a_name':name,'b_name':label})
        dataframe.to_csv("./G-Meta/new_data/{}/val.csv".format(dataset_source),sep=',')
        
        name=[]
        label=[]
        for idx in idx_test:
            name.append('0_{}'.format(idx))
            label.append(labels[idx].item())
            label_pkl['0_{}'.format(idx)]= labels[idx].item()
        dataframe = pd.DataFrame({'a_name':name,'b_name':label})
        dataframe.to_csv("./G-Meta/new_data/{}/test.csv".format(dataset_source),sep=',')
        
        np.save("./G-Meta/new_data/{}/features.npy".format(dataset_source),features.numpy())
        pkl.dump(label_pkl,open("./G-Meta/new_data/{}/label.pkl".format(dataset_source),'wb'))
        
        G = nx.from_numpy_matrix(sparse_mx.todense(),create_using=nx.DiGraph())
        #graph=[dgl.from_scipy(sparse_mx)]
        #graph=[dgl.from_networkx(G)]
        graph=[dgl.DGLGraph(G)]
        pkl.dump(graph,open('./G-Meta/new_data/{}/graph_dgl.pkl'.format(dataset_source),'wb'))
        
        print(1/0)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    return adj, features, labels, idx_train, idx_valid, idx_test, n1s, n2s, class_train_dict, class_test_dict


def neighborhoods_(adj, n_hops, use_cuda):
    """Returns the n_hops degree adjacency matrix adj."""
    # adj = torch.tensor(adj, dtype=torch.float)
    # adj=adj.to_dense()
    # print(type(adj))
    if use_cuda:
        adj = adj.cuda()
    # hop_adj = power_adj = adj

    # return (adj@(adj.to_dense())+adj).to_dense().cpu().numpy().astype(int)

    hop_adj = adj + torch.sparse.mm(adj, adj)

    hop_adj = hop_adj.to_dense()
    # hop_adj = (hop_adj > 0).to_dense()

    # for i in range(n_hops - 1):
    # power_adj = power_adj @ adj
    # prev_hop_adj = hop_adj
    # hop_adj = hop_adj + power_adj
    # hop_adj = (hop_adj > 0).float()

    hop_adj = hop_adj.cpu().numpy().astype(int)

    return (hop_adj > 0).astype(int)

    # return hop_adj.cpu().numpy().astype(int)


def neighborhoods(adj, n_hops, use_cuda):
    """Returns the n_hops degree adjacency matrix adj."""
    # adj = torch.tensor(adj, dtype=torch.float)
    # adj=adj.to_dense()
    # print(type(adj))
    if n_hops == 1:
        return adj.cpu().numpy().astype(int)

    if use_cuda:
        adj = adj.cuda()
    # hop_adj = power_adj = adj

    # for i in range(n_hops - 1):
    # power_adj = power_adj @ adj
    hop_adj = adj + adj @ adj
    hop_adj = (hop_adj > 0).float()

    np.save(hop_adj.cpu().numpy().astype(int), './neighborhoods_{}.npy'.format(dataset))

    return hop_adj.cpu().numpy().astype(int)


def InforNCE_Loss(anchor, sample, tau, all_negative=False, temperature_matrix=None):
    def _similarity(h1: torch.Tensor, h2: torch.Tensor):
        h1 = F.normalize(h1)
        h2 = F.normalize(h2)
        return h1 @ h2.t()

    assert anchor.shape[0] == sample.shape[0]

    pos_mask = torch.eye(anchor.shape[0], dtype=torch.float).cuda()
    neg_mask = 1. - pos_mask

    sim = _similarity(anchor, sample / temperature_matrix if temperature_matrix != None else sample) / tau
    exp_sim = torch.exp(sim) * (pos_mask + neg_mask)

    if not all_negative:
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
    else:
        log_prob = - torch.log(exp_sim.sum(dim=1, keepdim=True))

    loss = log_prob * pos_mask
    loss = loss.sum(dim=1) / pos_mask.sum(dim=1)

    # print(1)
    # return -loss[:10].mean()

    # print(sim)

    return -loss.mean(), sim


class Predictor(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout=0):
        super(Predictor, self).__init__()
        self.linear1 = Linear(nfeat, nhid)
        self.linear2 = Linear(nhid, nout)

    def forward(self, x):
        return self.linear2(self.linear1(x).relu())


parser = argparse.ArgumentParser()
parser.add_argument('--use_cuda', action='store_true', default=True, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
parser.add_argument('--epochs', type=int, default=2000,
                    help='Number of epochs to train.')
parser.add_argument('--test_epochs', type=int, default=2000,
                    help='Number of epochs to train.')
parser.add_argument('--pretrain_lr', type=float, default=0.005,
                    help='Initial learning rate.')

parser.add_argument('--weight_decay', type=float, default=5e-4,  # 5e-4
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden1', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--hidden2', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--pretrain_dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', default='dblp',
                    help='Dataset:Amazon_clothing/Amazon_eletronics/dblp')

args = parser.parse_args(args=[])

# args.use_cuda = torch.cuda.is_available()


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.use_cuda:
    torch.cuda.manual_seed(args.seed)

loss_f = nn.CrossEntropyLoss()
# Load data


N = 5
K = 3
n_hops = 1
avail_train_num_per_class = 5

fine_tune_steps = 40
fine_tune_lr = 0.1

args.epochs = 10000
args.test_epochs = 100

results=defaultdict(dict)

#for dataset in ['dblp','Amazon_eletronics','cora-full']:
for dataset in ['Amazon_eletronics']:
#for dataset in ['ogbn-arxiv']:

            # for dataset in ['dblp']:
            # for dataset in ['dblp']:
            # for dataset in ['cora-full',]:

    adj_sparse, features, labels, idx_train, idx_val, idx_test, n1s, n2s, class_train_dict, class_test_dict = load_data_pretrain(
        dataset)

    if dataset!='ogbn-arxiv':
        features -= features.min(0, keepdim=True)[0]
        features /= features.max(0, keepdim=True)[0]

    # print(adj_sparse[[0,1,2,3]])

    args.hidden1=features.shape[-1]
    
    
    
    adj = adj_sparse.to_dense()
    if dataset!='ogbn-arxiv':
        adj=adj.cuda()
    else:
        args.pretrain_dropout=0
        args.epochs=20000
        args.pretrain_lr=0.005
        fine_tune_steps = 40
        fine_tune_lr = 0.1
        #avail_train_num_per_class=100
        avail_train_num_per_class=avail_train_num_per_class*10

    
    
    
    for N in [5,10]:
        for K in [3]:


            print('don')
            print(dataset)
            print('N={},K={}'.format(N,K))

            # print(adj[:10,:10])

            # edge_index=[[one1,one2] for one1,one2 in zip(n1s,n2s)]
            edge_index = torch.LongTensor(np.array([n1s, n2s]))

            # total_neighbors=neighborhoods(adj=adj, n_hops=n_hops, use_cuda=True)
            # total_neighbors=adj.cpu().numpy()
            total_neighbors = json.load(open('./few_shot_data/neighbors/neighbors_{}.json'.format(dataset)))

            model = GCN_dense(nfeat=args.hidden1,
                              nhid=args.hidden2,
                              nclass=labels.max().item() + 1,
                              dropout=args.pretrain_dropout)

            phi_model = GCN_dense(nfeat=args.hidden1,
                                  nhid=args.hidden2,
                                  nclass=labels.max().item() + 1,
                                  dropout=args.pretrain_dropout)

            # generater=nn.Linear(args.hidden1, (args.hidden1+1)*args.hidden2*2+(args.hidden2+1)*args.hidden2*2)

            GCN_model = GCN_emb(nfeat=features.shape[1],
                                nhid=args.hidden1,
                                nclass=labels.max().item() + 1,
                                dropout=args.pretrain_dropout)

            classifier = Linear(args.hidden2, N)

            predictor = Predictor(args.hidden2, args.hidden2 * 2, args.hidden2)

            optimizer = optim.Adam([{'params': model.parameters()},
                                    {'params': GCN_model.parameters()}, {'params': classifier.parameters()},
                                    {'params': predictor.parameters()}, {'params': phi_model.parameters()}],
                                   lr=args.pretrain_lr, weight_decay=args.weight_decay)

            cos_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

            if args.use_cuda:
                model.cuda()
                GCN_model.cuda()
                features = features.cuda()
                #
                # generater=generater.cuda()
                # adj = adj.cuda()
                adj_sparse = adj_sparse.cuda()
                labels = labels.cuda()
                classifier = classifier.cuda()
                predictor = predictor.cuda()
                phi_model = phi_model.cuda()


            def pre_train(epoch, mode='train'):
                # classes=np.random.choice(list(range(len(class_train_dict))),N,replace=False)

                #emb_features = GCN_model(features, adj_sparse)
                emb_features=features

                target_idx = []
                target_new_idx = []
                target_graph_adj_and_feat = []
                support_target_graph_adj_and_feat = []

                pos_node_idx = []

                if mode == 'train':
                    class_dict = class_train_dict
                    for i in class_dict:  
                        class_dict[i] = class_dict[i][:avail_train_num_per_class]
                else:
                    class_dict = class_test_dict

                classes = np.random.choice(list(class_dict.keys()), N, replace=False).tolist()

                # for novel sample
                # classes.extend([-1]*N)

                for i in classes:

                    # sample from one specific class

                    pos_node_idx.extend(np.random.choice(class_dict[i], K, replace=False).tolist())

                    ############################################################################################
                    # build target subgraph
                    while True:
                        idx = np.random.choice(class_dict[i], 1, replace=False)[0]
                        # target_neighbors=total_neighbors[idx].nonzero()[0]
                        target_neighbors = total_neighbors[idx]

                        if len(target_neighbors) <= 1 or idx in pos_node_idx:
                            continue
                        else:
                            break

                    target_neighbors_2hop = []
                    for one in target_neighbors:
                        # target_neighbors_2hop.extend(total_neighbors[one].nonzero()[0])
                        target_neighbors_2hop.extend(total_neighbors[one])

                    if dataset=='ogbn-arxiv':target_neighbors_2hop=target_neighbors 

                        
                    target_neighbors = list(set(target_neighbors_2hop))

                    target_new_idx.append(target_neighbors.index(idx))
                    target_idx.append(idx)
                    target_graph_adj = adj[target_neighbors, :][:, target_neighbors].cuda()
                    # target_graph_edge_index= (target_graph_adj > 0).nonzero().t()

                    target_graph_feat = emb_features[target_neighbors]
                    # print(target_graph_feat.shape)
                    target_graph_adj_and_feat.append([target_graph_adj, target_graph_feat])
                    ############################################################################################

                # build support graph
                # pos_node_idx is a list containing NK nodes
                pos_graph_neighbors = torch.nonzero(adj[pos_node_idx, :].sum(0)).squeeze()

                random_graph_neighbors = torch.nonzero(
                    adj[np.random.choice(list(range(features.shape[0])), 10, replace=False).tolist(), :].sum(0)).squeeze()
                pos_graph_neighbors = torch.nonzero(adj[pos_graph_neighbors, :].sum(0)).squeeze()

                temp = pos_graph_neighbors.cpu().numpy().tolist() + random_graph_neighbors.cpu().numpy().tolist()

                # make sure the first NK nodes are labeled (support nodes)
                for idx in pos_node_idx:
                    while idx in temp:
                        temp.remove(idx)
                temp = pos_node_idx + temp

                pos_graph_neighbors = temp

                # pos_node_indices store the corresponding index of support nodes in the support graph
                # pos_node_indices.append([temp.index(one) for one in pos_node_idx])

                pos_graph_adj = adj[pos_graph_neighbors, :][:, pos_graph_neighbors].cuda()
                # pos_class_graph_adj=torch.eye(pos_graph_neighbors.shape[0]+1,dtype=torch.float).cuda()
                # pos_class_graph_adj[1:,1:]=pos_graph_adj
                # pos_graph_edge_index= (pos_class_graph_adj > 0).nonzero().t()

                pos_graph_feat = emb_features[pos_graph_neighbors]

                support_graph_adj_and_feat = [pos_graph_adj, pos_graph_feat]

                # POisson Learning
                #####################################################

                # attention=torch.exp(-cal_euclidean(pos_graph_feat)*500)

                attention = torch.exp(-cal_euclidean(pos_graph_feat) * 10)


                W = pos_graph_adj + attention

                D = torch.diag(W.sum(-1))
                L = D - W

                F = torch.zeros(N, N * K).cuda()
                for i in range(N):
                    F[i, i * K:(i + 1) * K] = 1
                y_bar = F.sum(-1) / F.shape[-1]

                B = torch.zeros(N, pos_graph_adj.shape[0]).cuda()
                B[:, :N * K] = F - y_bar.unsqueeze(-1)
                U = torch.zeros(pos_graph_adj.shape[0], N).cuda()

                T = 10 

                
                for i in range(T):
                    U = U + D.inverse().matmul(B.t() - L.matmul(U))

                # U=U.softmax(-1)

                U = U.softmax(-1)

                # print(U)
                # print(torch.argmax(U,-1))

                entropy = -(torch.log(U + 1e-9) * U).sum(-1)

                # print(entropy)

                # entropy_select_num = 20
                entropy_select_num = 10
                
                if dataset=='ogbn-arxiv':entropy_select_num=0
                
                threshold = 1.2
                U_idx_selected = []

                for i in range(N * K):
                    # if i in torch.argsort(entropy)[:entropy_select_num+N*K] or i < N * K:
                    U_idx_selected.append(i)
                for i in torch.argsort(entropy)[:entropy_select_num + N * K].cpu().numpy().tolist():
                    if len(U_idx_selected) == N * K + entropy_select_num: break
                    if i >= N * K:
                        U_idx_selected.append(i)

                U_labels_selected = torch.argmax(U[U_idx_selected], -1)
                for i in range(N):
                    U_labels_selected[i * K:(i + 1) * K] = i


                pseudo_acc = []
                for i, idx in enumerate([pos_graph_neighbors[U_idx] for U_idx in U_idx_selected]):
                    if i < N * K: continue
                    if classes[U_labels_selected[i]] == labels[idx]:
                        pseudo_acc.append(1.)
                    else:
                        pseudo_acc.append(0.)

                # build target subgraph for each support node (including selected nodes)
                ############################################################################################
                new_idx = []
                for idx in [pos_graph_neighbors[U_idx] for U_idx in U_idx_selected]:

                    target_neighbors = total_neighbors[idx]
                    target_neighbors_2hop = []
                    for one in target_neighbors:
                        # target_neighbors_2hop.extend(total_neighbors[one].nonzero()[0])
                        target_neighbors_2hop.extend(total_neighbors[one])

                    if dataset=='ogbn-arxiv':target_neighbors_2hop=target_neighbors 

                    
                    target_neighbors = list(set(target_neighbors_2hop))

                    new_idx.append(target_neighbors.index(idx))

                    target_graph_adj = adj[target_neighbors, :][:, target_neighbors].cuda()
                    # target_graph_edge_index= (target_graph_adj > 0).nonzero().t()

                    target_graph_feat = emb_features[target_neighbors]
                    # print(target_graph_feat.shape)
                    support_target_graph_adj_and_feat.append([target_graph_adj, target_graph_feat])
                ############################################################################################

                if mode == 'train':
                    model.train()
                    optimizer.zero_grad()
                else:
                    model.eval()

                gc1_w, gc1_b, gc2_w, gc2_b, w, b = model.gc1.weight, model.gc1.bias, model.gc2.weight, model.gc2.bias, classifier.weight, classifier.bias

                permute_emb = []
                for i, one in enumerate(support_target_graph_adj_and_feat):
                    sub_adj, sub_feat = one[0], one[1]
                    permute_adj, permute_feat = model.permute(sub_adj, sub_feat, 0.1)
                    permute_feat[new_idx[i]] = sub_feat[new_idx[i]]
                    permute_emb.append(phi_model(permute_feat, permute_adj)[new_idx[i]])  # .mean(0))
                
                permute_emb = torch.stack(permute_emb, 0)
                
                for j in range(fine_tune_steps):

                    ori_emb = []
                    
                    for i, one in enumerate(support_target_graph_adj_and_feat):
                        sub_adj, sub_feat = one[0], one[1]

                        ori_emb.append(model(sub_feat, sub_adj, gc1_w, gc1_b, gc2_w, gc2_b)[new_idx[i]])  # .mean(0))
                        



                    ori_emb = torch.stack(ori_emb, 0)
                    
                    loss_supervised = loss_f(classifier(ori_emb, w, b), U_labels_selected)
                    


                    # loss_reconstruction=((ori_emb/torch.norm(ori_emb,dim=-1,keepdim=True)-permute_emb/torch.norm(permute_emb,dim=-1,keepdim=True))+1e-9).square().sum()


                    loss_reconstruction = -cos_similarity(predictor(ori_emb), permute_emb).mean()
                    loss = loss_supervised + loss_reconstruction

                    grad = torch.autograd.grad(loss, [gc1_w, gc1_b, gc2_w, gc2_b, w, b])
                    gc1_w, gc1_b, gc2_w, gc2_b, w, b = list(
                        map(lambda p: p[1] - fine_tune_lr * p[0], zip(grad, [gc1_w, gc1_b, gc2_w, gc2_b, w, b])))

                    # print(grad)
                    if torch.isnan(grad[0]).sum() > 0:
                        print(grad)
                        print(1 / 0)

                query_labels = torch.tensor(list(range(N))).cuda()

                model.eval()
                ori_emb = []
                for i, one in enumerate(target_graph_adj_and_feat):
                    sub_adj, sub_feat = one[0], one[1]

                    ori_emb.append(model(sub_feat, sub_adj, gc1_w, gc1_b, gc2_w, gc2_b)[target_new_idx[i]])  # .mean(0))

                ori_emb = torch.stack(ori_emb, 0)
                logits = classifier(ori_emb, w, b)
                
                if np.random.rand()<0.01:print(logits)

                # print(ori_emb)

                loss = loss_f(logits, query_labels)

                if mode == 'train':
                    loss.backward()
                    optimizer.step()

                if epoch % 50 == 0 and mode == 'train':
                    print('Epoch: {:04d}'.format(epoch + 1),
                          'loss_train: {:.4f}'.format(loss.item()),  # 'loss_dis: {:.4f}'.format(dis_loss.item()),
                          'loss_super: {:.4f}'.format(loss_supervised.item()),
                          #'loss_recon: {:.4f}'.format(loss_reconstruction.item()),
                          'acc_train: {:.4f}'.format((torch.argmax(logits, -1) == query_labels).float().mean().item()),
                          'pseudo_acc: {:.4f} total_node: {}'.format(sum(pseudo_acc) / (len(pseudo_acc) + 1e-9),
                                                                     pos_graph_adj.shape[0]))

                return (torch.argmax(logits, -1) == query_labels).float().mean().item(), sum(pseudo_acc) / (len(pseudo_acc) + 1e-9)


            t_total = time.time()
            best_acc = 0
            pse=[]
            for epoch in range(args.epochs):
                acc_train, pse_acc_train=pre_train(epoch)
                #if epoch % 100 == 99:
                if epoch%500==0:pse.append(pse_acc_train)
                if epoch % 500 == 499:
                    acc = 0
                    pse_acc=0
                    for epoch_test in range(args.test_epochs):
                        acc_,pse_acc_=pre_train(epoch_test, mode='test')
                        acc+=acc_
                        pse_acc+=pse_acc_

                        del acc_,pse_acc_

                    acc/=args.test_epochs
                    pse_acc/=args.test_epochs

                    if acc > best_acc: best_acc = acc

                    print('Final Test Acc: {:.4f}  Best Acc: {:.4f} Pse_test acc: {:.4f} '.format(acc, best_acc, pse_acc))

                    del acc, pse_acc


                    
            
            
            results[dataset]['{}-way {}-shot'.format(N,K)]=[best_acc,pse]       
            
            json.dump(results,open('./SSL_result_ogbn_r+r_nofirstGCN_funetunelr0.1_step40_lr0.005_2hop_5ratio_3shot','w'))
                 

             
                    
            print("Optimization Finished!")
            print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

            torch.save(model.state_dict(),
                       './saved_models/GIN_{}_{}_epochs_class_ego_contrast.pth'.format(dataset, args.epochs))
            
            del model,phi_model,GCN_model

    del adj

keys = ['5-way 3-shot', '5-way 5-shot', '10-way 3-shot', '10-way 5-shot']

for name in names:
    print(name.replace('_','-')+'&' + '&'.join(keys)+'\\\\')
    print('Result' + '&' + '&'.join(
        ['{:.2f}'.format(one * 100) for one in [results[name][key] for key in keys]]) + '\\\\')     


