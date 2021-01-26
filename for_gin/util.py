import networkx as nx
import numpy as np
import random
import torch
import torch.nn as nn
import pickle
from sklearn.model_selection import StratifiedKFold

class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0

        self.max_neighbor = 0

def load_data(dataset, degree_as_tag):
    '''
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    '''
    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}

    with open('dataset/%s/%s.txt' % (dataset, dataset), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags, node_features = [], []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features, node_feature_flag = None, False

            assert len(g) == n

            g_list.append(S2VGraph(g, l, node_tags))

    #add labels and edge_mat       
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        g.edge_mat = torch.LongTensor(edges).transpose(0,1)

    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree).values())

    #Extracting unique tag labels   
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]:i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = torch.zeros(len(g.node_tags), len(tagset))
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1


    print('# classes: %d' % len(label_dict))
    print('# maximum node tag: %d' % len(tagset))
    print("# data: %d" % len(g_list))

    return g_list, len(label_dict)

def separate_data(graph_list, seed, fold_idx):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)

    labels = [graph.label for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]

    return train_graph_list, test_graph_list


#def contrastive_loss_labelwise(args, batch_idx_by_label, train_idx_by_label, hidden_feats):
#    '''
#    hidden feats must be normalized.
#    '''
#    assert(len(batch_idx_by_label) == len(train_idx_by_label))
#    hidden_feats = nn.functional.normalize(hidden_feats, dim=1)
#
#    loss = 0
#    for i in batch_idx_by_label:
#        if(len(batch_idx_by_label[i]) == 0):
#            continue
#        q, k = hidden_feats[batch_idx_by_label[i]], hidden_feats[train_idx_by_label[i]]
#        # k_neg = hidden_feats[list(set([i for i in range(len_train_graphs)]) - set(train_idx_by_label[i]))]
#        l_pos = torch.sum(torch.exp(torch.mm(q, k.transpose(0,1))/args.temperature), dim=1)
#        l_neg = torch.sum(torch.exp(torch.mm(q, hidden_feats.transpose(0,1))/args.temperature), dim=1)
#        # print('part loss', l_pos/l_neg)
#        loss += torch.sum(-1.0*torch.log(l_pos/l_neg))
#    return loss/args.batch_size


def contrastive_loss_labelwise_winslide(args, batch_idx_by_label, train_idx_by_label, hidden_feats, queue):
    '''
    hidden feats must be normalized.
    '''
    assert(len(batch_idx_by_label) == len(train_idx_by_label))
    hidden_feats = nn.functional.normalize(hidden_feats, dim=1)

    loss = 0
    for i in batch_idx_by_label:
        if(len(batch_idx_by_label) == 0):
            continue
        q, k = hidden_feats[batch_idx_by_label[i]], queue[train_idx_by_label[i]]
        l_pos = torch.sum(torch.exp(torch.mm(q, k.transpose(0,1))/args.temperature), dim=1)
        l_neg = torch.sum(torch.exp(torch.mm(q, queue.transpose(0,1))/args.temperature), dim=1)
        loss += torch.sum(-1.0*torch.log(l_pos/l_neg))
    return loss/args.batch_size


def contrastive_loss_samplewise_winslide(args, hidden_feats, queue):
    '''
    hidden feats must be normalized.
    '''
    hidden_feats = nn.functional.normalize(hidden_feats, dim=1)

    loss = 0
    # for i in batch_idx_by_label:

    q = hidden_feats
    #print('q size', q.size())
    l_pos = torch.sum(torch.exp(q*q/args.temperature), dim=1)
    l_neg = torch.sum(torch.exp(torch.mm(q, queue.transpose(0,1))/args.temperature), dim=1)
    # print('two part size',l_pos.size(), l_neg.size())
    loss = torch.sum(-1.0*torch.log(l_pos/l_neg))
    return loss/args.batch_size


@torch.no_grad()
def momentum_update(encoder_q, encoder_k, m=0.999):
    """
    encoder_k = m * encoder_k + (1 - m) encoder_q
    """        
    for param_q, param_k in zip(encoder_q.parameters(), encoder_k.parameters()):
        param_k.data = param_k.data * m + param_q.data * (1. - m)

    return encoder_k


def dequeue_and_enqueue(hidden_batch_feats, selected_batch_idx, queue):
    '''
    update memory bank by batch window slide; hidden_batch_feats must be normalized
    '''
    assert(hidden_batch_feats.size()[1] == queue.size()[1])


    queue[selected_batch_idx] = nn.functional.normalize(hidden_batch_feats,dim=1)
    return queue


def dequeue_and_enqueue_multiLayer(hidden_feats_dict, selected_batch_idx, queue):
    '''
    update memory bank by batch window slide; hidden_batch_feats must be normalized
    '''
    assert(len(hidden_feats_dict) == len(queue))

    for i in range(len(queue)):
        queue[i][selected_batch_idx] = nn.functional.normalize(hidden_feats_dict[i],dim=1)
    return queue


def to_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x

