import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import copy
from tqdm import tqdm

from util import load_data, separate_data, to_cuda, contrastive_loss_labelwise_winslide, dequeue_and_enqueue_multiLayer, momentum_update
from models.graphcnn_pooled_multilayer import GraphCNN

criterion = nn.CrossEntropyLoss()

def train(args, model_q, model_k, device, train_graphs, num_classes, optimizer, queue, epoch):
    train_idx_by_label = {}
    for i in range(num_classes):
        train_idx_by_label[i] = [idx for idx in range(len(train_graphs)) if train_graphs[idx].label == i]

    model_q.train()
    model_k.train()

    pbar = tqdm(range(args.iters_per_epoch), unit='batch')

    celoss_accum, contraloss_accum = 0,0
    for pos in pbar:
        selected_batch_idx = np.random.permutation(len(train_graphs))[:args.batch_size]
        batch_graph = [train_graphs[idx] for idx in selected_batch_idx]

        output, _ = model_q(batch_graph)

        batch_labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)
        #compute cross-entropy loss
        celoss = criterion(output, batch_labels)
        # update memory bank
        _, hidden_batch_feats = model_k(batch_graph)
        queue = dequeue_and_enqueue_multiLayer(hidden_batch_feats, selected_batch_idx, queue)
        # compute label-wise contrastive loss
        batch_idx_by_label = {}
        for i in range(num_classes):
            batch_idx_by_label[i] = [idx for idx in range(len(batch_graph)) if batch_graph[idx].label == i]
        
        contraloss = 0.0
        for layer in hidden_batch_feats:
            contraloss += contrastive_loss_labelwise_winslide(args, batch_idx_by_label, train_idx_by_label, 
                                                            hidden_batch_feats[layer], queue[layer].detach().clone())
        # momentum update model k
        model_k = momentum_update(model_q, model_k, m=0.999)
        # backprop
        loss = celoss + args.contraloss_weight * contraloss
        optimizer.zero_grad()
        loss.backward()         
        optimizer.step()

        celoss_accum += celoss.detach().cpu().numpy()
        contraloss_accum += contraloss.detach().cpu().numpy()


        #report
        pbar.set_description('epoch: %d' % (epoch))

    avg_celoss = celoss_accum/args.iters_per_epoch
    avg_contraloss = contraloss_accum/args.iters_per_epoch
    print("training celoss: %4f, contraloss: %4f" % (avg_celoss, avg_contraloss))
    
    return avg_celoss, avg_contraloss, queue

###pass data to model with minibatch during testing to avoid memory overflow (does not perform backpropagation)
def pass_data_iteratively(model, graphs, minibatch_size = 64):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model([graphs[j] for j in sampled_idx])[0].detach())
    return torch.cat(output, 0)

def test(args, model, device, train_graphs, test_graphs, epoch):
    model.eval()

    output = pass_data_iteratively(model, train_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in train_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_train = correct / float(len(train_graphs))

    output = pass_data_iteratively(model, test_graphs)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_test = correct / float(len(test_graphs))

    print("accuracy train: %f test: %f" % (acc_train, acc_test))

    return acc_train, acc_test

def main():
    # Training settings
    # Note: Hyper-parameters need to be tuned in order to obtain results reported in the paper.
    parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--dataset', type=str, default="MUTAG", help='name of dataset')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training')
    parser.add_argument('--iters_per_epoch', type=int, default=50, help='number of iterations per each epoch')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--seed', type=int, default=0, help='random seed for splitting the dataset into 10')
    parser.add_argument('--fold_idx', type=int, default=0, help='the index of fold in 10-fold validation. Should be less then 10.')
    parser.add_argument('--num_layers', type=int, default=5, help='number of layers INCLUDING the input one')
    parser.add_argument('--num_mlp_layers', type=int, default=2, help='number of layers for MLP EXCLUDING the input one. 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=64, help='number of hidden units')
    parser.add_argument('--final_dropout', type=float, default=0.5, help='final layer dropout')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average"], help='Pooling for over nodes in a graph: sum or average')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average", "max"], help='Pooling for over neighboring nodes: sum, average or max')
    parser.add_argument('--learn_eps', action="store_true", help='Whether to learn epsilon weighting forcenter nodes. Do not affect training accuracy.')
    parser.add_argument('--degree_as_tag', action="store_true", help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
    parser.add_argument('--filename', type = str, default = "", help='output file')
    parser.add_argument('--contraloss_weight', type=float, default=0.5, help='The weight of contrastive loss term.')
    parser.add_argument('--temperature', type=float, default=0.07, help='The temperature for contrastive loss.')
    args = parser.parse_args()

    #set up seeds and gpu device
    torch.manual_seed(0)
    np.random.seed(0)    
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    graphs, num_classes = load_data(args.dataset, args.degree_as_tag)

    Val_results, Train_results, Train_celoss, Train_contraloss = {},{},{},{}
    ##10-fold cross validation. Conduct an experiment on the fold specified by args.fold_idx.
    for fold_idx in range(10):
        train_graphs, test_graphs = separate_data(graphs, args.seed, fold_idx)

        model_q = GraphCNN(args.num_layers, args.num_mlp_layers, train_graphs[0].node_features.shape[1], args.hidden_dim, num_classes, args.final_dropout, args.learn_eps, args.graph_pooling_type, args.neighbor_pooling_type, device).to(device)
        model_k = copy.deepcopy(GraphCNN(args.num_layers, args.num_mlp_layers, train_graphs[0].node_features.shape[1], args.hidden_dim, num_classes, args.final_dropout, args.learn_eps, args.graph_pooling_type, args.neighbor_pooling_type, device)).to(device)
        for param_q, param_k in zip(model_q.parameters(), model_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        optimizer = optim.Adam(model_q.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

        # create queue
        model_q.train()
        queue = model_q(train_graphs)[1]
        for i in queue:
            queue[i] = F.normalize(queue[i], dim=1)

        val_acc, train_acc, train_celoss, train_contraloss = [],[],[],[]
        for epoch in range(1, args.epochs + 1):
            scheduler.step()

            avg_celoss, avg_contraloss, queue = train(args, model_q, model_k, device, train_graphs, num_classes, optimizer, queue, epoch)
            acc_train, acc_test = test(args, model_q, device, train_graphs, test_graphs, epoch)
            val_acc.append(acc_test)
            train_acc.append(acc_train)
            train_celoss.append(avg_celoss)
            train_contraloss.append(avg_contraloss)

        Val_results[fold_idx], Train_results[fold_idx], Train_celoss[fold_idx] = val_acc, train_acc, train_celoss
        Train_contraloss[fold_idx] = train_contraloss
 

if __name__ == '__main__':
    main()
