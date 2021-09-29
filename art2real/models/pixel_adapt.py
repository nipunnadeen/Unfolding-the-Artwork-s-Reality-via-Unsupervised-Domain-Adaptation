from __future__ import print_function

import os
from os.path import join

# Import from torch
import torch
import torch.optim as optim
import torch.utils.data as data

# from ..models.models import get_model
from models.fcn import Discriminator, AddaDataLoader
# from models.feature_loss import models
# from ..data.data_loader import load_data
# from .test_task_net import test
from torch.autograd import Variable
from util.util import make_variable

from options.train_options import TrainOptions
from data import create_dataset
from models import create_model

models = {}


def get_model(name, num_cls=10, **args):
    net = models[name](num_cls=num_cls, **args)
    if torch.cuda.is_available():
        net = net.cuda()
    return net


def train_epoch(loader, net, opt_net, epoch):
    log_interval = 100  # specifies how often to display
    net.train()
    for batch_idx, (data, target) in enumerate(loader):

        # make data variables
        data = make_variable(data, requires_grad=False)
        target = make_variable(target, requires_grad=False)

        # zero out gradients
        opt_net.zero_grad()

        # forward pass
        score = net(data)
        loss = net.criterion(score, target)

        # backward pass
        loss.backward()

        # optimize classifier and representation
        opt_net.step()

        # Logging
        if batch_idx % log_interval == 0:
            print('[Train] Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(loader.dataset),
                       100. * batch_idx / len(loader), loss.item()), end="")
            pred = score.data.max(1)[1]
            correct = pred.eq(target.data).cpu().sum()
            acc = correct.item() / len(pred) * 100.0
            print('  Acc: {:.2f}'.format(acc))


def train(model, num_cls, batch=128,
          lr=1e-4, betas=(0.9, 0.999), weight_decay=0, epoch=0):
    """Train a classification net and evaluate on test set."""

    # Setup GPU Usage
    if torch.cuda.is_available():
        kwargs = {'num_workers': 1, 'pin_memory': True}
    else:
        kwargs = {}

    ############
    # Load Net #
    ############
    net = get_model(model, num_cls=num_cls)
    print('-------Training net--------')
    print(net)

    ############################
    # Load train and test data #
    ############################

    # train_data = load_data(data, 'train', batch=batch,
    #                        rootdir=datadir, num_channels=net.num_channels,
    #                        image_size=net.image_size, download=True, kwargs=kwargs)
    #
    # loader = AddaDataLoader(net.transform, data, 128, False, 2)
    # train_data = loader

    # train_data = torch.utils.data.DataLoader(data, batch_size=batch,
    #                                          shuffle='train', **kwargs)
    datadir = '/content/drive/My Drive/Colab Notebooks/Art2Real/art2real/datasets/portrait2photo/trainA'
    train_data = torch.utils.data.DataLoader(datadir, batch_size=batch, shuffle=True, **kwargs)

    # test_data = load_data(data, 'test', batch=batch,
    #                       rootdir=datadir, num_channels=net.num_channels,
    #                       image_size=net.image_size, download=True, kwargs=kwargs)

    ###################
    # Setup Optimizer #
    ###################
    opt_net = optim.Adam(net.parameters(), lr=lr, betas=betas,
                         weight_decay=weight_decay)

    #########
    # Train #
    #########
    train_epoch(train_data, net, opt_net, epoch)
    # print('Training {} model for {}'.format(model, data))
    # for epoch in range(num_epoch):
    #     train_epoch(train_data, net, opt_net, epoch)

    # ########
    # # Test #
    # ########
    # if test_data is not None:
    #     print('Evaluating {}-{} model on {} test set'.format(model, data, data))
    #     test(test_data, net)

    return net
