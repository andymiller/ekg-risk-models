"""
Model classes for predicting with ECGS
"""
from __future__ import division
import os, shutil, time, sys, pyprind, pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchnet as tnt
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.nn import init
import numpy as np
do_cuda = torch.cuda.is_available()

# base models
#s#ys.path.append("../common/")
#import util, evaluate
#from model import base
from ekgmodels import base


##############################
# Classifier and Regression  #
##############################


class EKGResNetClassifier(base.Model):
    def __init__(self, **kwargs): 
        super(EKGResNetClassifier, self).__init__(**kwargs)
        self.net = EKGResNet(**kwargs)
        self.loss = base.NanBCEWithLogitsLoss(size_average=True)
        self.is_continuous = False

    def lossfun(self, data, target):
        logit = self.forward(data)
        pred_loss = self.loss(logit, target)
        return torch.mean(pred_loss), logit

    def forward(self, data):
        return self.net(data)

    def fit(self, Xdata, Ydata, **kwargs):
        Xtrain, Xval, Xtest = Xdata
        Ytrain, Yval, Ytest = Ydata
        self.fit_res = base.fit_mlp(self, Xtrain, Xval, Xtest,
                                    Ytrain, Yval, Ytest, 
                                    **kwargs)
        return self.fit_res


class EKGDeepWideResNetClassifier(EKGResNetClassifier):
    """ wide and deep EKG resnet.  Expects the last `dim_wide` dimensions
        to be linearly added into the final prediction. 
    """
    def __init__(self, **kwargs):
        # first, make the EKGResnet have 100 outputs
        self.h_dim     = kwargs.get("h_dim")
        self.n_outputs = kwargs.get("total_n_outputs")
        self.dim_wide  = kwargs.get("dim_wide")
        kwargs['n_outputs'] = self.h_dim
        super(EKGDeepWideResNetClassifier, self).__init__(**kwargs)
        self.wide_out = nn.Linear(self.h_dim + self.dim_wide, self.n_outputs, bias=True)

    def forward(self, data):
        """ this module takes in a batch_sz x (C x T + dim_wide) """
        # ekg transform
        last_out = data[:, -self.dim_wide:]
        ekg_shape = (data.shape[0], self.net.n_channels, self.net.n_samples)
        ekg_data = data[:, :-self.dim_wide].contiguous()
        ekg_data = ekg_data.view(ekg_shape)

        # wide + EKG representation
        zout = torch.cat([self.net(ekg_data), last_out], 1)
        return self.wide_out(zout)


class EKGResNetRegression(base.Model):
    def __init__(self, **kwargs): 
        super(EKGResNetRegression, self).__init__(**kwargs)
        self.net = EKGResNet(**kwargs)
        self.loss = nn.MSELoss(size_average=True)
        self.is_continuous = True

    def lossfun(self, data, target):
        logit = self.net(data)
        pred_loss = self.loss(logit, target)
        return torch.mean(pred_loss), logit

    def fit(self, Xdata, Ydata, **kwargs):
        Xtrain, Xval, Xtest = Xdata
        Ytrain, Yval, Ytest = Ydata
        self.fit_res = base.fit_mlp(self, Xtrain, Xval, Xtest,
                                    Ytrain, Yval, Ytest, 
                                    **kwargs)
        return self.fit_res


####################################################################
# Implementation of the conv resnet model from Rajpurkar, Hannun,  #
# Haghpanahi Bourn and Ng 2017                                     #
####################################################################

class EKGResNet(base.Model):
    """ Residual network, described in Rajpurkar et al, 2017

    Args:
      - n_channels: number of leads used (batches are size bsz x n_channels x n_samples)
      - n_samples : how long each tracing is
      - n_outputs : number of {0,1} features to predict
      - num_rep_blocks: how deep the network is --- how many repeated internal
          resnet blocks
    """
    def __init__(self, **kwargs): 
        super(EKGResNet, self).__init__(**kwargs)
        n_channels = kwargs.get("n_channels")
        n_samples = kwargs.get("n_samples")
        n_outputs = kwargs.get("n_outputs")
        num_rep_blocks = kwargs.get("num_rep_blocks", 16) #depth of the network
        kernel_size = kwargs.get("kernel_size", 16)
        self.verbose = kwargs.get("verbose", False)

        # store dimension info
        self.n_channels, self.n_samples, self.n_outputs = \
          n_channels, n_samples, n_outputs

        # track signal length so you can appropriately set the fully connected layer
        self.seq_len = n_samples

        # set up first block
        self.block0 = nn.Sequential(
            nn.Conv1d(in_channels=n_channels, out_channels=64,
                      kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(64),
            nn.ReLU())
        self.seq_len += 1

        # input --- output is bsz x 64 x ceil(n_samples/2)
        self.block1 = BlockOne(in_channels=64, out_channels=64, kernel_size=16)
        self.seq_len = self.seq_len // 2

        # repeated block
        self.num_rep_blocks = num_rep_blocks
        self.num_layers     = 3 + 2*num_rep_blocks + 1 # each rep block is 2
        in_features = 64
        num_max_pool = 0

        self.rep_blocks = nn.ModuleList()
        for l in range(num_rep_blocks):
            # determine number of output features
            out_features = in_features
            b = RepBlock(in_channels=in_features, out_channels=out_features,
                         kernel_size=16)
            self.rep_blocks.append(b)
            in_features = out_features

            # count how many features are removed
            if l % 2 == 0:
                num_max_pool += 1

        # update seq_len
        self.seq_len = self.seq_len // (2**num_max_pool)
        self.num_last_active = int(self.seq_len * out_features)

        # output block
        self.block_out = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.ReLU())
        print(self.num_last_active, n_outputs)
        self.fc_out = nn.Linear(self.num_last_active, n_outputs)

        print("net thinks it will have")
        print("  seq_len     :", self.seq_len)
        print("  last active :", self.num_last_active)

        # maxpool, used throughout
        self.mp = nn.MaxPool1d(2)

    def init_params(self):
        for p in self.parameters():
            if p.ndimension() >= 2:
                init.kaiming_normal(p)
            else:
                init.normal(p, mean=0., std=.02)

    def printif(self, *args):
        if self.verbose:
            print(" ".join([str(a) for a in args]))

    def forward(self, x):
        # first conv layer
        x = self.block0(x)
        self.printif("block0 out:", x.size())

        # block 1 --- subsample input and maxpool input for residual
        xmp = self.mp(x)
        x   = self.block1(x[:,:,::2])
        x   += xmp
        self.printif("block1 out:", x.size())

        # repeated blocks, apply
        for l, blk in enumerate(self.rep_blocks):
            if l % 2 == 0:
                xmp, xin = self.mp(x), x[:,:,:-1:2]
            else:
                xmp, xin = x, x
            x = blk(xin.contiguous())
            self.printif(l, "  repblock shape; xtmp shape ", x.shape, xmp.shape)
            x += xmp

        # fully connected out layer
        self.printif("before block_out", x.shape)
        x = self.block_out(x.contiguous())
        self.printif("after block_out", x.shape)
        self.printif("self.num_last_active", self.num_last_active)
        x = x.view(x.size()[0], self.num_last_active)
        self.printif("after view", x.shape)
        x = self.fc_out(x)
        self.printif("after fc_out", x.shape)
        return x


class BlockOne(nn.Module):
    """ Resnet, first convolutional block """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        """ if subsample is true, then we subsample input and maxpool
        the residual term """
        super(BlockOne, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                      kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout2d())
        self.conv = nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, padding=kernel_size//2)

    def forward(self, x):
        out = self.block(x)
        out = out[:,:,:-1]
        out = self.conv(out)
        out = out[:,:,:-1]
        return out


class RepBlock(nn.Module):
    """ Resnet, Repeated Convolutional Block"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(RepBlock, self).__init__()

        self.block1 = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                      kernel_size=kernel_size, padding=kernel_size//2))

        self.block2 = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout2d())

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = out[:,:,:-2]
        return out



############################################################
# Simpler convnet models with varying architectures        #
############################################################
#<<<<<<< Updated upstream
model_a = { 'n_filter_list': [16, 32, 64, 128],
            'kernel_sizes' : [20, 10, 10, 10],
            'strides'      : [2, 1, 1, 1],
            'pool_sizes'   : [5, 4, 2, 2],
            'full_sizes'   : [100, 10] }

model_b = { 'n_filter_list': [8,  16, 32, 64],
            'kernel_sizes' : [50, 10, 10, 6],
            'strides'      : [2,   2, 1, 1],
            'pool_sizes'   : [4,   4, 2, 1],
            'full_sizes'   : [100, 10] }

model_c = { 'n_filter_list': [16, 32, 64],
            'kernel_sizes' : [20, 10, 10],
            'strides'      : [2,   2,  1],
            'pool_sizes'   : [5,   5,  3],
            'full_sizes'   : [100, 10] }

model_d = { 'n_filter_list': [16, 32, 64, 128],
            'kernel_sizes' : [20, 10, 10, 10],
            'strides'      : [2, 1, 1, 1],
            'pool_sizes'   : [5, 4, 2, 2],
            'full_sizes'   : [100, 100] }

conv_model_dict = { 'conv_a' : model_a,
                    'conv_b' : model_b,
                    'conv_c' : model_c,
                    'conv_d' : model_d }

class EcgConvNet(nn.Module):
    def __init__(self, n_channels, n_samples, n_outputs, **kwargs):

        """Creates a convolutional neural network to predict a
        categorical output from a multi-channel ecg

        Args:
            n_channels : number of ecg channels (one for each lead)
            n_samples  : number of samples for each lead --- 1499 for the long ones
            n_outputs  : number of categorical outputs for the network (number of classes to predict)
        """
        super(EcgConvNet, self).__init__()

        n_filter_list = kwargs.get('n_filter_list', [16, 32, 64, 128])
        kernel_sizes  = kwargs.get('kernel_sizes', [20, 10, 10, 10])
        strides       = kwargs.get('strides'     , [2, 1, 1, 1])
        pool_sizes    = kwargs.get('pool_sizes'  , [5, 4, 2, 2])
        full_sizes    = kwargs.get('full_sizes'  , [100, 10])
        full_dropout  = kwargs.get('full_dropout', 0.)
        conv_dropout  = kwargs.get('conv_dropout', 0.)
        do_batchnorm  = kwargs.get('do_batchnorm', True)
        self.do_batchnorm = do_batchnorm

        # set up convolutional layers
        self.conv_list = []
        n_in = n_channels
        for i, (n_out, ksz, stride, psz) in enumerate(zip(n_filter_list, kernel_sizes, strides, pool_sizes)):
            cl = nn.Conv1d(in_channels  = n_in,
                           out_channels = n_out,
                           kernel_size  = ksz,
                           stride       = stride)
            mp = nn.MaxPool1d(kernel_size = psz)
            bn = nn.BatchNorm1d(n_out)
            self.add_module("conv_bn_%d"%i, bn)
            self.add_module("conv_%d"%i, cl)
            self.conv_list.append((cl, mp, bn))
            n_in = n_out

        self.conv_dropout = nn.Dropout2d(p=conv_dropout)
        self.n_filter_out = n_out

        # set up full layers
        self.full_list = []
        layer_sizes = [n_out] + full_sizes
        for i, (nin, nout) in enumerate(zip(layer_sizes[:-1],
                                            layer_sizes[1:])):
            lay = nn.Linear(nin, nout)
            bn  = nn.BatchNorm1d(nout)
            self.add_module("full_%d"%i, lay)
            self.add_module("full_bn_%d"%i, bn)
            self.full_list.append((lay, bn))

        self.full_dropout = nn.Dropout(p=full_dropout)

        # output layer -- takes the last feature and transforms it to output size
        self.output  = nn.Linear(full_sizes[-1], n_outputs)
        self.verbose = False

    def apply_conv_layers(self, x):
        """ run convolution layers """
        # convolutional layers -- cl, max pool and batch norm
        for i, (cl, mp, bn) in enumerate(self.conv_list):
            if self.verbose:
                print("x %d"%i, x.size())

            # apply convolution
            x = cl(x)

            # apply batchnorm
            if self.do_batchnorm:
                x = bn(x)

            # apply dropout to last layer
            if i == len(self.conv_list)-1 and self.conv_dropout.p > 0:
                x = self.conv_dropout(x)

            # apply max pool + relu
            x = F.relu(mp(x))

        if self.verbose:
            print("post conv size: ", x.size())

        # reshape for fully connected layers
        x = x.view(x.size()[0], self.n_filter_out)
        return x

    def apply_full_layers(self, x):
        for i, (lay, bn) in enumerate(self.full_list):
            if self.verbose:
                print("x fc %d"%i, x.size())

            x = lay(x)
            if self.do_batchnorm:
                x = bn(x)

            #x = F.relu(x)
            if i != len(self.full_list)-1:
                x = F.relu(x)

            if self.full_dropout.p > 0:  # and i < len(self.full_list)-1:
                x = self.full_dropout(x)

        return x

    def features(self, x):
        # apply conv, and full layers
        x = self.apply_conv_layers(x)
        x = self.apply_full_layers(x)
        return x

    def forward(self, x):
        # make features w/ conv layers and full layers, apply output
        x = self.features(x)
        x = self.output(x)
        return x


######################
# Training Functions #
######################

#def train_model(data_train, data_val, data_test, output_dir,
#                target_names, **kwargs):
#    """ Run multiple epochs, reducing learning rate based on validation
#    set performance
#    """
#    batch_size    = kwargs.get("batch_size", 256)
#    depth         = kwargs.get("depth", "medium")
#    masked        = kwargs.get("masked", True)
#    learning_rate = kwargs.get("learning_rate", 1e-4)
#    epochs        = kwargs.get("epochs", 100)
#    print_freq    = kwargs.get("print_freq", 10)
#    if not os.path.exists(output_dir):
#        os.makedirs(output_dir)
#
#    # create train/val/test datasets and loaders
#    Xtrain, Ytrain = data_train
#    Xval,   Yval   = data_val
#    Xtest,  Ytest  = data_test
#    train_dataset = torch.utils.data.TensorDataset(
#        torch.from_numpy(Xtrain).float(), torch.from_numpy(Ytrain).float())
#    val_dataset = torch.utils.data.TensorDataset(
#        torch.from_numpy(Xval).float(), torch.from_numpy(Yval).float())
#    test_dataset = torch.utils.data.TensorDataset(
#        torch.from_numpy(Xtest).float(), torch.from_numpy(Ytest).float())
#
#    train_loader = torch.utils.data.DataLoader(train_dataset,
#        batch_size=batch_size, shuffle=True, pin_memory=True, sampler=None)
#    val_loader = torch.utils.data.DataLoader(val_dataset,
#        batch_size=batch_size, shuffle=False, pin_memory=True)
#    test_loader = torch.utils.data.DataLoader(test_dataset,
#        batch_size=batch_size, shuffle=False, pin_memory=True)
#
#    # create model
#    if depth=="shallow":
#        num_rep_blocks = 1
#    elif depth=="medium":
#        num_rep_blocks = 8
#    elif depth=="deep":
#        num_rep_blocks = 16
#    model = EcgResNet(n_channels = Xtrain.shape[1],
#                      n_samples  = Xtrain.shape[2],
#                      n_outputs  = Ytrain.shape[1],
#                      num_rep_blocks = num_rep_blocks)
#
#    # Define loss function (criterion) and optimizer
#    if masked:
#        criterion = MaskedBCELoss(size_average=True)
#    else:
#        criterion = nn.BCELoss(size_average=True) # nn.CrossEntropyLoss()
#
#    # create optimizer
#    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#    if do_cuda:
#        torch.backends.cudnn.enabled=False  #for some reason this needs to be turned off?
#        model.cuda()
#
#    # track per epoch Train+Val loss --- record test losses on best models
#    train_loss_per_epoch = []
#    val_loss_per_epoch   = []
#
#    all_auc_meter_list = []
#    all_auc_list = []
#    all_loss_list = []
#
#    best_accuracy = 0
#    best_auc = 0
#    current_auc = 0
#
#    train_all_auc_list = []
#    train_all_loss_list = []
#    train_best_auc = 0
#    train_current_auc = 0
#
#    # Train and evaluate
#    prev_valloss, best_valloss = float("inf"), float("inf")
#    for epoch in range(epochs):
#
#        # run adam for one epoch
#        epoch_dict = train_epoch(epoch, train_loader, model, criterion, optimizer,
#            target_names=target_names, print_freq=print_freq, masked=masked)
#        train_loss_per_epoch.append(epoch_dict['loss'])
#
#        # evaluate on validation set
#        curr_valloss, val_preddf = compute_loss(val_loader, model, criterion,
#            target_names=target_names, masked=masked)
#        val_loss_per_epoch.append(curr_valloss)
#        print("--- validation prediction statistics (valloss=%2.5f) --- "%curr_valloss)
#        print(val_preddf[['num_obs', 'frac_pos', 'frac_neg',
#                          'accs', 'aucs', 'f1', 'precision', 'recall']])
#
#        # save training statistics
#        epoch_state = {'train_loss_per_epoch': train_loss_per_epoch,
#                       'val_loss_per_epoch'  : val_loss_per_epoch,
#                       'train_preddf'        : epoch_dict['preddf'],
#                       'val_preddf'          : val_preddf }
#        with open(os.path.join(output_dir,
#                               "epoch_state_%d.pkl"%epoch), 'wb') as f:
#            pickle.dump(epoch_state, f)
#
#        # check validation against best model
#        if curr_valloss < best_valloss:
#            print("\n... found best validation loss: saving model")
#            best_valloss = curr_valloss
#
#            # evaluate on test
#            test_loss, test_preddf = compute_loss(test_loader, model,
#                criterion, target_names=target_names, masked=masked)
#
#            print("--- test prediction statistics ---")
#            print(test_preddf[['num_obs', 'frac_pos', 'frac_neg',
#                               'accs', 'aucs', 'f1', 'precision', 'recall']])
#
#            # save model + prediction information
#            torch.save({'epoch'       : epoch,
#                        'state_dict'  : model.state_dict(),
#                        'test_preddf' : test_preddf,
#                        'train_preddf': epoch_dict['preddf'],
#                        'val_preddf'  : val_preddf,
#                        'test_loss'   : test_loss },
#                        f=os.path.join(output_dir, "model.best.pth.tar"))
#
#        if curr_valloss >= prev_valloss:
#            new_lr = util.decay_lr(optimizer, .1)
#            print(" ... decayed learning rate to %2.6f"%new_lr)
#
#        prev_valloss = curr_valloss
#
#    return model
#
#
#def train_epoch(epoch, train_loader, model, criterion,
#                optimizer, target_names, print_freq, masked):
#    """ trains a single EPOCH """
#    batch_time = util.AverageMeter()
#    data_time = util.AverageMeter()
#    losses = util.AverageMeter()
#    accuracy = util.AverageMeter()
#    auc_meter = tnt.meter.AUCMeter()
#
#    # track prediction probabilities
#    probs, ytrue = [], []
#
#    # Switch to train mode
#    model.train()
#
#    end = time.time()
#    for i, (xin, yout) in enumerate(train_loader):
#        # Measure data loading time
#        data_time.update(time.time() - end)
#
#        # Compute output
#        if masked:
#            data, target, mask = prep_batch(xin, yout, do_cuda, return_mask=True)
#            output = model(data)
#            loss = criterion(F.sigmoid(output), target, mask)
#        else:
#            data, target = prep_batch(xin, yout, do_cuda, return_mask=False)
#            output = model(data)
#            loss = criterion(F.sigmoid(output), target)
#
#        # Measure accuracy and record loss
#        losses.update(loss.data[0], xin.size(0))
#
#        # Compute gradient and do gradient update
#        optimizer.zero_grad()
#        loss.backward()
#        optimizer.step()
#
#        # Measure elapsed time
#        batch_time.update(time.time() - end)
#        end = time.time()
#
#        # track training statistics
#        probs.append(F.sigmoid(output).data.cpu().numpy())
#        ytrue.append(target.data.cpu().numpy())
#
#        # AUC update
#        #auc_meter.add(output.max(1)[1].data.cpu().numpy()[:,0], target.cpu().numpy())
#        if (i % print_freq == 0) or (i == len(train_loader)-1):
#            print('Epoch: [{0}] ({1}/{2})\t'
#                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
#                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                  'Accuracy {accuracy.val:.3f} ({accuracy.avg:.3f})'.format(
#                    epoch, i, len(train_loader), batch_time=batch_time,
#                    data_time=data_time, loss=losses, accuracy=accuracy))
#
#            print(" --- train prediction statistics --- ")
#            preddf = evaluate.report_prediction_stats(
#                np.row_stack(ytrue), np.row_stack(probs), target_names, num_boot=1)
#            print(preddf[['num_obs', 'frac_pos', 'frac_neg',
#                          'accs', 'aucs', 'f1', 'precision', 'recall']])
#
#    epoch_dict = {"preddf": preddf, "loss": losses.avg}
#    return epoch_dict
#
#
#def prep_batch(X, Y, do_cuda, return_mask=False):
#    data, target = Variable(X), Variable(Y)
#    if do_cuda:
#        data, target = data.cuda(async=True), target.float().cuda(async=True)
#
#    if return_mask:
#        mask = Variable(torch.ByteTensor(np.array(~np.isnan(Y), dtype=np.int)))
#        if do_cuda:
#            mask = mask.cuda(async=True)
#        return data, target, mask
#
#    return data, target
#
#
#def compute_loss(data_loader, model, criterion, target_names, masked):
#    model.eval()
#    losses = util.AverageMeter()
#    probs, ytrue = [], []
#    for xin, yout in pyprind.prog_bar(data_loader):
#        if masked:
#            data, target, mask= prep_batch(xin, yout, do_cuda, return_mask=True)
#            output = model(data)
#            loss   = criterion(F.sigmoid(output), target, mask)
#        else:
#            data, target = prep_batch(xin, yout, do_cuda)
#            output = model(data)
#            loss   = criterion(F.sigmoid(output), target)
#
#        # Measure accuracy and record loss
#        losses.update(loss.data[0], xin.size(0))
#
#        # track training statistics
#        probs.append(F.sigmoid(output).data.cpu().numpy())
#        ytrue.append(target.data.cpu().numpy())
#
#    preddf = evaluate.report_prediction_stats(
#        np.row_stack(ytrue), np.row_stack(probs), target_names, num_boot=200)
#    return losses.avg, preddf


