from __future__ import print_function

import os
import argparse
import time

import torch
import numpy as np
from torch.optim import Adam, lr_scheduler
from torch.nn import functional as F

from PyTorchBayesianCNN import data
import utils
import metrics
import config_bayesian as cfg
from models.PartiallyBayesianModels.PartiallyBayesian3Conv3FC import FirstLayerBBB3Conv3FC, LastLayerBBB3Conv3FC, FirstandLastLayerBBB3Conv3FC
from models.PartiallyBayesianModels.PartiallyBayesianAlexNet import FirstLayerBBBAlexNet, LastLayerBBBAlexNet, FirstandLastLayerBBBAlexNet
from models.PartiallyBayesianModels.PartiallyBayesianLeNet import FirstLayerBBBLeNet, LastLayerBBBLeNet, FirstandLastLayerBBBLeNet, LastBlockBBBLeNet

# CUDA settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def getModel(net_type, stochasticity, inputs, outputs, priors, layer_type, activation_type):
    if (net_type == 'lenet'):
        if (stochasticity == 'first'):
            return FirstLayerBBBLeNet(outputs, inputs, priors, layer_type, activation_type)
        elif (stochasticity == 'last'):
            return LastLayerBBBLeNet(outputs, inputs, priors, layer_type, activation_type)
        elif (stochasticity == 'firstlast'):
            return FirstandLastLayerBBBLeNet(outputs, inputs, priors, layer_type, activation_type)
    elif (net_type == 'alexnet'):
        if (stochasticity == 'first'):
            return FirstLayerBBBAlexNet(outputs, inputs, priors, layer_type, activation_type)
        elif (stochasticity == 'last'):
            return LastLayerBBBAlexNet(outputs, inputs, priors, layer_type, activation_type)
        elif (stochasticity == 'firstlast'):
            return FirstandLastLayerBBBAlexNet(outputs, inputs, priors, layer_type, activation_type)
    elif (net_type == '3conv3fc'):
        if (stochasticity == 'first'):
            return FirstLayerBBB3Conv3FC(outputs, inputs, priors, layer_type, activation_type)
        elif (stochasticity == 'last'):
            return LastLayerBBB3Conv3FC(outputs, inputs, priors, layer_type, activation_type)
        elif (stochasticity == 'firstlast'):
            return FirstandLastLayerBBB3Conv3FC(outputs, inputs, priors, layer_type, activation_type)
    else:
        raise ValueError('Network should be either [LeNet / AlexNet / 3Conv3FC')


def train_model(net, optimizer, criterion, trainloader, num_ens=1, beta_type=0.1, epoch=None, num_epochs=None):
    net.train()
    training_loss = 0.0
    accs = []
    kl_list = []
    for i, (inputs, labels) in enumerate(trainloader, 1):

        optimizer.zero_grad()

        inputs, labels = inputs.to(device), labels.to(device)
        outputs = torch.zeros(inputs.shape[0], net.num_classes, num_ens).to(device)

        kl = 0.0
        for j in range(num_ens):
            net_out, _kl = net(inputs)
            kl += _kl
            outputs[:, :, j] = F.log_softmax(net_out, dim=1)
        
        kl = kl / num_ens
        kl_list.append(kl.item())
        log_outputs = utils.logmeanexp(outputs, dim=2)

        beta = metrics.get_beta(i-1, len(trainloader), beta_type, epoch, num_epochs)
        loss = criterion(log_outputs, labels, kl, beta)
        loss.backward()
        optimizer.step()

        accs.append(metrics.acc(log_outputs.data, labels))
        training_loss += loss.cpu().data.numpy()
    return training_loss/len(trainloader), np.mean(accs), np.mean(kl_list)


def validate_model(net, criterion, validloader, num_ens=1, beta_type=0.1, epoch=None, num_epochs=None):
    """Calculate ensemble accuracy and NLL Loss and ECE"""
    net.train()
    valid_loss = 0.0
    accs = []
    all_log_outputs = []
    targets = []

    for i, (inputs, labels) in enumerate(validloader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = torch.zeros(inputs.shape[0], net.num_classes, num_ens).to(device)
        kl = 0.0
        for j in range(num_ens):
            net_out, _kl = net(inputs)
            kl += _kl
            outputs[:, :, j] = F.log_softmax(net_out, dim=1).data

        log_outputs = utils.logmeanexp(outputs, dim=2)
        all_log_outputs.append(log_outputs)
        targets.append(labels)

        beta = metrics.get_beta(i-1, len(validloader), beta_type, epoch, num_epochs)
        valid_loss += criterion(log_outputs, labels, kl, beta).item()
        accs.append(metrics.acc(log_outputs, labels))
        
    all_log_outputs = torch.cat(all_log_outputs)
    targets = torch.cat(targets)
    return valid_loss/len(validloader), np.mean(accs), metrics.ece(all_log_outputs, targets)

def test_model(net, criterion, validloader, num_ens=1, beta_type=0.1, epoch=None, num_epochs=None):
    """Calculate ensemble accuracy and NLL Loss and ECE"""
    net.eval()
    valid_loss = 0.0
    accs = []
    all_log_outputs = []
    targets = []

    for i, (inputs, labels) in enumerate(validloader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = torch.zeros(inputs.shape[0], net.num_classes, num_ens).to(device)
        kl = 0.0
        for j in range(num_ens):
            net_out, _kl = net(inputs)
            kl += _kl
            outputs[:, :, j] = F.log_softmax(net_out, dim=1).data

        log_outputs = utils.logmeanexp(outputs, dim=2)
        all_log_outputs.append(log_outputs)
        targets.append(labels)

        beta = metrics.get_beta(i-1, len(validloader), beta_type, epoch, num_epochs)
        valid_loss += criterion(log_outputs, labels, kl, beta).item()
        accs.append(metrics.acc(log_outputs, labels))
        
    all_log_outputs = torch.cat(all_log_outputs)
    targets = torch.cat(targets)
    return F.nll_loss(all_log_outputs, targets), np.mean(accs), metrics.ece(all_log_outputs, targets)


def run(dataset, net_type, stochasticity):
    
    # Hyper Parameter settings
    layer_type = cfg.layer_type
    activation_type = cfg.activation_type
    priors = cfg.priors

    train_ens = cfg.train_ens
    valid_ens = cfg.valid_ens
    n_epochs = cfg.n_epochs
    lr_start = cfg.lr_start
    num_workers = cfg.num_workers
    valid_size = cfg.valid_size
    batch_size = cfg.batch_size
    beta_type = cfg.beta_type

    trainset, testset, inputs, outputs = data.getDataset(dataset)
    train_loader, valid_loader, test_loader = data.getDataloader(
        trainset, testset, valid_size, batch_size, num_workers)
    net = getModel(net_type, stochasticity, inputs, outputs, priors, layer_type, activation_type).to(device)

    ckpt_dir = f'checkpoints/{dataset}/bayesian'
    ckpt_name = f'checkpoints/{dataset}/bayesian/model_{net_type}_{layer_type}_{activation_type}.pt'

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    criterion = metrics.ELBO(len(trainset)).to(device)
    optimizer = Adam(net.parameters(), lr=lr_start)
    lr_sched = lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, verbose=True)
    valid_loss_max = np.Inf
    start = time.time()
    for epoch in range(n_epochs):  # loop over the dataset multiple times

        train_loss, train_acc, train_kl = train_model(net, optimizer, criterion, train_loader, num_ens=train_ens, beta_type=beta_type, epoch=epoch, num_epochs=n_epochs)
        valid_loss, valid_acc, valid_ece = validate_model(net, criterion, valid_loader, num_ens=valid_ens, beta_type=beta_type, epoch=epoch, num_epochs=n_epochs)
        lr_sched.step(valid_loss)

        print('Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tValidation Loss: {:.4f} \tValidation Accuracy: {:.4f} \tValidation ECE: {:.4f} \ttrain_kl_div: {:.4f}'.format(
            epoch, train_loss, train_acc, valid_loss, valid_acc, valid_ece, train_kl))

        # save model if validation accuracy has increased
        if valid_loss <= valid_loss_max:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_max, valid_loss))
            torch.save(net.state_dict(), ckpt_name)
            valid_loss_max = valid_loss
    end = time.time()
    print('Total training time: {:.2f} secs'.format(end - start))
    test_loss, test_acc, test_ece = test_model(net, criterion, test_loader, num_ens=5)
    print('\tTest Loss: {:.4f} \tTest Accuracy: {:.4f} \tTest ECE: {:.4f}'.format(test_loss, test_acc, test_ece))
    return end - start, test_loss, test_acc, test_ece

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "PyTorch Bayesian Model Training")
    parser.add_argument('--net_type', default='lenet', type=str, help='model')
    parser.add_argument('--stochasticity', default='first', type=str, help='[first/last/firstlast/block]')
    parser.add_argument('--dataset', default='MNIST', type=str, help='dataset = [MNIST/CIFAR10/CIFAR100]')
    args = parser.parse_args()

    run(args.dataset, args.net_type, args.stochasticity)
