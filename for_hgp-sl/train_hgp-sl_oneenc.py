import argparse
import glob
import os
import time
import pickle
import torch
import torch.nn.functional as F
import numpy as np
from models import Model
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from util import contrastive_loss_labelwise_winslide, dequeue_and_enqueue_HGPSL

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--sample_neighbor', type=bool, default=True, help='whether sample neighbors')
parser.add_argument('--sparse_attention', type=bool, default=True, help='whether use sparse attention')
parser.add_argument('--structure_learning', type=bool, default=True, help='whether perform structure learning')
parser.add_argument('--pooling_ratio', type=float, default=0.5, help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.0, help='dropout ratio')
parser.add_argument('--lamb', type=float, default=1.0, help='trade-off parameter')
parser.add_argument('--dataset', type=str, default='PROTEINS', help='DD/PROTEINS/NCI1/')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
parser.add_argument('--epochs', type=int, default=1000, help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')
parser.add_argument('--contraloss_weight', type=float, default=0.5, help='The weight of contrastive loss term.')
parser.add_argument('--temperature', type=float, default=0.07, help='The temperature for contrastive loss.')

args = parser.parse_args()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

dataset = TUDataset(os.path.join('data', args.dataset), name=args.dataset, use_node_attr=True)

args.num_classes = dataset.num_classes
args.num_features = dataset.num_features

num_training = int(len(dataset) * 0.8)
num_val = int(len(dataset) * 0.1)
num_test = len(dataset) - (num_training + num_val)


def train(model, optimizer, queue, train_loader, val_loader, train_idx_by_label):
    min_loss = 1e10
    patience_cnt = 0
    val_loss_values = []
    best_epoch = 0

    t = time.time()
    model.train()

    val_acc, train_acc, train_celoss, train_contraloss= [],[],[],[]
    for epoch in range(args.epochs):
        celoss_train = 0.0
        contraloss_train = 0.0
        correct = 0
        start_ptr = 0
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(args.device)
            out, hidden_feats_dict = model(data)
            celoss = F.nll_loss(out, data.y)

            # update memory bank
            queue = dequeue_and_enqueue_HGPSL(hidden_feats_dict, start_ptr, start_ptr+len(data.y), queue)
            start_ptr += len(data.y)
            # compute label-wise contrastive loss
            batch_idx_by_label = {}
            for i in range(args.num_classes):
                batch_idx_by_label[i] = [idx for idx in range(len(data.y)) if data.y[idx] == i]

            contraloss = 0.0
            for layer in hidden_feats_dict:
                contraloss += contrastive_loss_labelwise_winslide(args, batch_idx_by_label, train_idx_by_label, 
                                                                hidden_feats_dict[layer], queue[layer].detach().clone())

            loss = celoss + args.contraloss_weight*contraloss
            loss.backward()
            optimizer.step()
            celoss_train += celoss.item()
            contraloss_train += contraloss
            pred = out.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
        acc_train = correct / len(train_loader.dataset)
        acc_val, loss_val = compute_test(val_loader)
        print('Epoch: {:04d}'.format(epoch + 1), 'celoss_train: {:.6f}'.format(celoss_train), 'contraloss_train: {:.6f}'.format(contraloss_train),
              'acc_train: {:.6f}'.format(acc_train), 'loss_val: {:.6f}'.format(loss_val),
              'acc_val: {:.6f}'.format(acc_val), 'time: {:.6f}s'.format(time.time() - t))

        train_celoss.append(celoss_train)
        train_contraloss.append(contraloss_train)
        train_acc.append(acc_train)
        val_acc.append(acc_val)
        val_loss_values.append(loss_val)
        torch.save(model.state_dict(), '{}.winpth'.format(epoch))
        if val_loss_values[-1] < min_loss:
            min_loss = val_loss_values[-1]
            best_epoch = epoch
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt == args.patience:
            break

        files = glob.glob('*.winpth')
        for f in files:
            epoch_nb = int(f.split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(f)

    files = glob.glob('*.winpth')
    for f in files:
        epoch_nb = int(f.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(f)
    print('Optimization Finished! Total time elapsed: {:.6f}'.format(time.time() - t))

    return best_epoch, train_celoss, train_contraloss, train_acc, val_acc, val_loss_values


def compute_test(loader):
    model.eval()
    correct = 0.0
    loss_test = 0.0
    for data in loader:
        data = data.to(args.device)
        out, _ = model(data)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss_test += F.nll_loss(out, data.y).item()
    return correct / len(loader.dataset), loss_test


if __name__ == '__main__':
    Train_celoss, Train_contraloss, Train_acc, Val_loss, Val_acc = {},{},{},{},{}
    for i in range(10):

        # prepare data
        training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])
        train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=False)
        val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

        train_idx_by_label = {}
        for i in range(args.num_classes):
            train_idx_by_label[i] = [idx for idx in range(num_training) if training_set[idx].y == i]

        model = Model(args).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # Model training
        queue = {1:F.normalize(torch.randn(num_training, args.nhid), dim=1).to(args.device),
         2:F.normalize(torch.randn(num_training, args.nhid//2), dim=1).to(args.device),
         3:F.normalize(torch.randn(num_training, args.num_classes), dim=1).to(args.device)}

        best_model, train_celoss, train_contraloss, train_acc, val_acc, val_loss = train(model, optimizer, queue, train_loader, val_loader, train_idx_by_label)
        Train_celoss[i], Train_contraloss[i], Train_acc[i], Val_loss[i], Val_acc[i] = train_celoss, train_contraloss, train_acc, val_loss, val_acc
        # Restore best model for test set
        model.load_state_dict(torch.load('{}.winpth'.format(best_model))) 
        test_acc, test_loss = compute_test(test_loader)
        Test_loss.append(test_loss)
        Test_acc.append(test_acc)
        
        show_results += np.array(val_acc)
        files = glob.glob('*.winpth')
        for f in files:
            epoch_nb = int(f.split('.')[0])
            if epoch_nb == best_model:
                os.remove(f)

    

