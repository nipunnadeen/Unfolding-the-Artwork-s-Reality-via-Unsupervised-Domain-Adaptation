import os
import os.path
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from models.fcn import Discriminator
from util.util import make_variable

models = {}


def get_model(name, num_cls=10, **args):
    net = models[name](num_cls=num_cls, **args)
    if torch.cuda.is_available():
        net = net.cuda()
    return net


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def supervised_loss(score, label, weights=None):
    loss_fn_ = torch.nn.NLLLoss(weight=weights, size_average=True,
                                ignore_index=255)
    label = label.long()
    loss = loss_fn_(F.log_softmax(score, dim=1), label)
    return loss


class Feature_loss():
    def __init__(self, lr, momentum, cls_weights, num_cls, lsgan, lambda_d, lambda_g, train_discrim_only,
                 weights_discrim, weights_shared, discrim_feat, im_s, im_t, label_s, label_t, net):

        self.weights = None
        self.featureLoss = None

        # losses_dis = deque(maxlen=100)
        targetSup = 1

        np.random.seed(1337)
        torch.manual_seed(1337)

        # # can be simplified the following condition
        # if weights_shared:
        #     net_src = net  # shared weights
        # else:
        #     net_src = net
        #     net_src.eval()

        odim = 1 if lsgan else 2
        idim = num_cls if not discrim_feat else 4096

        discriminator = Discriminator(input_dim=idim, output_dim=odim,
                                      pretrained=not (weights_discrim is None),
                                      weights_init=weights_discrim).cuda()
        # Class weighted loss?
        if cls_weights is not None:
            weights = np.loadtxt(cls_weights)
        else:
            weights = None

        # setup optimizers
        opt_dis = torch.optim.SGD(discriminator.parameters(), lr=lr,
                                  momentum=momentum, weight_decay=0.0005)
        opt_rep = torch.optim.SGD(net.parameters(), lr=lr,
                                  momentum=momentum, weight_decay=0.0005)

        num_update_g = 0
        # losses_super_s = deque(maxlen=100)
        # losses_super_t = deque(maxlen=100)
        # losses_rep = deque(maxlen=100)
        # accuracies_dom = deque(maxlen=100)

        ###########################
        # 1. Setup Data Variables #
        ###########################
        im_s = make_variable(im_s, requires_grad=False)
        labelSource = make_variable(label_s, requires_grad=False)
        im_t = make_variable(im_t, requires_grad=False)
        labelTarget = make_variable(label_t, requires_grad=False)

        # # zero gradients for optimizer
        # opt_dis.zero_grad()
        # opt_rep.zero_grad()
        #
        # # extract features
        # if discrim_feat:
        #     score_s, feat_s = net_src(im_s)
        #     f_s = Variable(feat_s.data, requires_grad=False)
        # else:
        #     score_s = Variable(net_src(im_s).data, requires_grad=False)
        #     f_s = score_s
        # dis_score_s = discriminator(f_s)
        #
        # if discrim_feat:
        #     score_t, feat_t = net(im_t)
        #     f_t = Variable(feat_t.data, requires_grad=False)
        # else:
        #     score_t = Variable(net(im_t).data, requires_grad=False)
        #     f_t = score_t
        # dis_score_t = discriminator(f_t)
        #
        # dis_pred_concat = torch.cat((dis_score_s, dis_score_t))
        #
        # # prepare real and fake labels
        # batch_t, _, h, w = dis_score_t.size()
        # batch_s, _, _, _ = dis_score_s.size()
        # dis_label_concat = make_variable(
        #     torch.cat(
        #         [torch.ones(batch_s, h, w).long(),
        #          torch.zeros(batch_t, h, w).long()]
        #     ), requires_grad=False)
        #
        # # compute loss for discriminator
        # loss_dis = supervised_loss(dis_pred_concat, dis_label_concat)
        # (lambda_d * loss_dis).backward()
        # losses_dis.append(loss_dis.item())
        #
        # # optimize discriminator
        # opt_dis.step()
        #
        # # compute discriminator acc
        # pred_dis = torch.squeeze(dis_pred_concat.max(1)[1])
        # dom_acc = (pred_dis == dis_label_concat).float().mean().item()
        # accuracies_dom.append(dom_acc * 100.)

        ###########################
        # Optimize Target Network #
        ###########################

        # dom_acc_thresh = 55

        # if not train_discrim_only and np.mean(accuracies_dom) > dom_acc_thresh:
        if not train_discrim_only:

            num_update_g += 1
            if num_update_g % 1 == 0:
                print('Updating G with adversarial loss ({:d} times)'.format(num_update_g))

            # zero out optimizer gradients
            opt_dis.zero_grad()
            opt_rep.zero_grad()

            # extract features
            if discrim_feat:
                scoreTarget, feat_t = net(im_t)
                f_t = feat_t
            else:
                scoreTarget = net(im_t)
                f_t = scoreTarget

            dis_scoreTarget = discriminator(f_t)

            # create fake label
            batch, _, h, w = dis_scoreTarget.size()
            target_dom_fake_t = make_variable(torch.ones(batch, h, w).long(),
                                              requires_grad=False)

            # compute loss for target net
            loss_gan_t = supervised_loss(dis_scoreTarget, target_dom_fake_t)

            self.featureLoss = lambda_g * loss_gan_t

            (lambda_g * loss_gan_t).backward()
            # losses_rep.append(loss_gan_t.item())

            # optimize target net
            opt_rep.step()

        # if (not train_discrim_only) and weights_shared and (np.mean(accuracies_dom) > dom_acc_thresh):
        if (not train_discrim_only) and weights_shared:

            print('Updating G using source supervised loss.')

            # zero out optimizer gradients
            opt_dis.zero_grad()
            opt_rep.zero_grad()

            # extract features

            scoreSource = net(im_s)
            scoreTarget = net(im_t)

            # if discrim_feat:
            #     scoreSource, _ = net(im_s)
            #     scoreTarget, _ = net(im_t)
            # else:
            #     scoreSource = net(im_s)
            #     scoreTarget = net(im_t)

            loss_supervised_s = supervised_loss(scoreSource, labelSource, weights=weights)
            loss_supervised_t = supervised_loss(scoreTarget, labelTarget, weights=weights)
            loss_supervised = loss_supervised_s

            if targetSup:
                loss_supervised += loss_supervised_t

            self.featureLoss = loss_supervised
            loss_supervised.backward()

            # losses_super_s.append(loss_supervised_s.item())
            #
            # losses_super_t.append(loss_supervised_t.item())

            # optimize target net
            opt_rep.step()

    def compute_feature_loss(self):
        return self.featureLoss

    def get_weight(self):
        return self.weights
