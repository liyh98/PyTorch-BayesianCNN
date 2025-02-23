from __future__ import print_function

import os
import argparse
import time

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler

from PyTorchBayesianCNN import data
import utils
import metrics
import config_frequentist as cfg
from models.NonBayesianModels.AlexNet import AlexNet
from models.NonBayesianModels.LeNet import LeNet
from models.NonBayesianModels.ThreeConvThreeFC import ThreeConvThreeFC

# CUDA settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def getModel(net_type, inputs, outputs):
    if (net_type == 'lenet'):
        return LeNet(outputs, inputs)
    elif (net_type == 'alexnet'):
        return AlexNet(outputs, inputs)
    elif (net_type == '3conv3fc'):
        return ThreeConvThreeFC(outputs, inputs)
    else:
        raise ValueError('Network should be either [LeNet / AlexNet / 3Conv3FC')


def train_model(net, optimizer, criterion, train_loader):
    train_loss = 0.0
    net.train()
    accs = []
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)
        accs.append(metrics.acc(output.detach(), target))
    return train_loss, np.mean(accs)


def validate_model(net, criterion, valid_loader):
    valid_loss = 0.0
    net.eval()
    accs = []
    for data, target in valid_loader:
        data, target = data.to(device), target.to(device)
        output = net(data)
        loss = criterion(output, target)
        valid_loss += loss.item()*data.size(0)
        accs.append(metrics.acc(output.detach(), target))
    return valid_loss, np.mean(accs)

def test_model(net, criterion, valid_loader):
    valid_loss = 0.0
    net.eval()
    accs = []
    targets = []
    outputs = []
    for data, target in valid_loader:
        data, target = data.to(device), target.to(device)
        output = net(data)
        targets.append(target)
        outputs.append(F.log_softmax(output.detach()))
        loss = criterion(output, target)
        valid_loss += loss.item()*data.size(0)
        accs.append(metrics.acc(output.detach(), target))
    targets = torch.cat(targets)
    outputs = torch.cat(outputs)
    return F.nll_loss(outputs, targets), np.mean(accs), metrics.ece(outputs, targets)

def run(dataset, net_type):

    # Hyper Parameter settings
    n_epochs = cfg.n_epochs
    lr = cfg.lr
    num_workers = cfg.num_workers
    valid_size = cfg.valid_size
    batch_size = cfg.batch_size

    trainset, testset, inputs, outputs = data.getDataset(dataset)
    train_loader, valid_loader, test_loader = data.getDataloader(
        trainset, testset, valid_size, batch_size, num_workers)
    net = getModel(net_type, inputs, outputs).to(device)

    ckpt_dir = f'checkpoints/{dataset}/frequentist'
    ckpt_name = f'checkpoints/{dataset}/frequentist/model_{net_type}.pt'

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=lr)
    lr_sched = lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, verbose=True)
    valid_loss_min = np.Inf
    start = time.time()
    for epoch in range(1, n_epochs+1):

        train_loss, train_acc = train_model(net, optimizer, criterion, train_loader)
        valid_loss, valid_acc = validate_model(net, criterion, valid_loader)
        lr_sched.step(valid_loss)

        train_loss = train_loss/len(train_loader.dataset)
        valid_loss = valid_loss/len(valid_loader.dataset)
            
        print('Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tValidation Loss: {:.4f} \tValidation Accuracy: {:.4f}'.format(
            epoch, train_loss, train_acc, valid_loss, valid_acc))
        
        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min, valid_loss))
            torch.save(net.state_dict(), ckpt_name)
            valid_loss_min = valid_loss
    end = time.time()
    print('Total training time: {:.2f} secs'.format(end - start))
    test_loss, test_acc, test_ece = test_model(net, criterion, test_loader)
    print('\tTest Loss: {:.4f} \tTest Accuracy: {:.4f} \tTest ECE: {:.4f}'.format(test_loss, test_acc, test_ece))
    return end - start, test_loss, test_acc, test_ece

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "PyTorch Frequentist Model Training")
    parser.add_argument('--net_type', default='lenet', type=str, help='model')
    parser.add_argument('--dataset', default='MNIST', type=str, help='dataset = [MNIST/CIFAR10/CIFAR100]')
    args = parser.parse_args()

    run(args.dataset, args.net_type)
