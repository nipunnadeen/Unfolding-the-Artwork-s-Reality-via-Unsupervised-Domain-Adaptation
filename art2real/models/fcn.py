
import numpy as np
import torch
import torch.nn.functional as F
# import torchvision
from torch import nn
from torch.autograd import Variable
from torch.nn import init
from torch.utils import model_zoo
# from torchvision.models import vgg

import os.path

from PIL import Image
import torch.utils.data

# from models.data_loader import get_transform_dataset
# from models.tranformImage import augment_collate
from options.train_options import TrainOptions


class Discriminator(nn.Module):
    def __init__(self, input_dim=4096, output_dim=2, pretrained=False, weights_init=''):
        super().__init__()
        dim1 = 1024 if input_dim==4096 else 512
        dim2 = int(dim1/2)
        self.D = nn.Sequential(
            nn.Conv2d(input_dim, dim1, 1),
            nn.Dropout2d(p=0.5),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim1, dim2, 1),
            nn.Dropout2d(p=0.5),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim2, output_dim, 1)
            )

        if pretrained and weights_init is not None:
            self.load_weights(weights_init)

    def forward(self, x):
        d_score = self.D(x)
        return d_score

    def load_weights(self, weights):
        print('Loading discriminator weights')
        self.load_state_dict(torch.load(weights))


# class Discriminator(nn.Module):
#     """Discriminator model for source domain."""
#
#     def __init__(self, input_dims, hidden_dims, output_dims):
#         """Init discriminator."""
#         super(Discriminator, self).__init__()
#
#         self.restored = False
#
#         self.layer = nn.Sequential(
#             nn.Linear(input_dims, hidden_dims),
#             nn.ReLU(),
#             nn.Linear(hidden_dims, hidden_dims),
#             nn.ReLU(),
#             nn.Linear(hidden_dims, output_dims),
#             nn.LogSoftmax()
#         )
#
#     def forward(self, input):
#         """Forward the discriminator."""
#         out = self.layer(input)
#         return out


class AddaDataLoader(object):
    # def __init__(self, net_transform, dataset, rootdir, downscale, crop_size=None,
    #              batch_size=1, shuffle=False, num_workers=2, half_crop=None):
    def __init__(self, dataset, downscale, batch_size=1, shuffle=False, num_workers=2):
        opt = TrainOptions().parse()
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.dataset = dataset
        self.downscale = downscale
        # self.crop_size = crop_size
        # self.half_crop = half_crop
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        assert len(self.dataset) == 2, 'Requires two datasets: source, target'
        # sourcedir = os.path.join(rootdir, self.dataset[0])
        # targetdir = os.path.join(rootdir, self.dataset[1])
        # self.source = get_transform_dataset(self.dataset[0], sourcedir,
        #                                     net_transform, downscale)
        # self.target = get_transform_dataset(self.dataset[1], targetdir,
        #                                     net_transform, downscale)
        sourcedir = os.path.join("/content/drive/My Drive/Colab Notebooks/Art2Real/art2real/datasets/trainA", self.dataset[0])
        targetdir = os.path.join("/content/drive/My Drive/Colab Notebooks/Art2Real/art2real/datasets/trainB", self.dataset[1])
        self.source = torch.load(sourcedir, map_location=str(self.device))
        self.target = torch.load(targetdir, map_location=str(self.device))

        print('Source length:', len(self.source), 'Target length:', len(self.target))
        self.n = max(len(self.source), len(self.target))  # make sure you see all images
        self.num = 0
        self.set_loader_src()
        self.set_loader_tgt()

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        # if self.num % len(self.iters_src) == 0:
        #     print('restarting source dataset')
        #     self.set_loader_src()
        # if self.num % len(self.iters_tgt) == 0:
        #     print('restarting target dataset')
        #     self.set_loader_tgt()

        img_src, label_src = next(self.iters_src)
        img_tgt, label_tgt = next(self.iters_tgt)

        self.num += 1
        return img_src, img_tgt, label_src, label_tgt

    def __len__(self):
        return min(len(self.source), len(self.target))

    def set_loader_src(self):
        batch_size = self.batch_size
        shuffle = self.shuffle
        num_workers = self.num_workers
        # if self.crop_size is not None:
        #     collate_fn = lambda batch: augment_collate(batch, crop=self.crop_size,
        #                                                halfcrop=self.half_crop, flip=True)
        # else:
        #     collate_fn = torch.utils.data.dataloader.default_collate

        collate_fn = torch.utils.data.dataloader.default_collate
        self.loader_src = torch.utils.data.DataLoader(self.source,
                                                      batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                                      collate_fn=collate_fn, pin_memory=True)
        self.iters_src = iter(self.loader_src)

    def set_loader_tgt(self):
        batch_size = self.batch_size
        shuffle = self.shuffle
        num_workers = self.num_workers
        # if self.crop_size is not None:
        #     collate_fn = lambda batch: augment_collate(batch, crop=self.crop_size,
        #                                                halfcrop=self.half_crop, flip=True)
        # else:
        #     collate_fn = torch.utils.data.dataloader.default_collate
        collate_fn = torch.utils.data.dataloader.default_collate
        self.loader_tgt = torch.utils.data.DataLoader(self.target,
                                                      batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                                      collate_fn=collate_fn, pin_memory=True)
        self.iters_tgt = iter(self.loader_tgt)




# class LeNetEncoder(nn.Module):
#     """LeNet encoder model for ADDA."""
#
#     def __init__(self):
#         """Init LeNet encoder."""
#         super(LeNetEncoder, self).__init__()
#
#         self.restored = False
#
#         self.encoder = nn.Sequential(
#             # 1st conv layer
#             # input [1 x 28 x 28]
#             # output [20 x 12 x 12]
#             nn.Conv2d(1, 20, kernel_size=5),
#             nn.MaxPool2d(kernel_size=2),
#             nn.ReLU(),
#             # 2nd conv layer
#             # input [20 x 12 x 12]
#             # output [50 x 4 x 4]
#             nn.Conv2d(20, 50, kernel_size=5),
#             nn.Dropout2d(),
#             nn.MaxPool2d(kernel_size=2),
#             nn.ReLU()
#         )
#         self.fc1 = nn.Linear(50 * 4 * 4, 500)
#
#     def forward(self, input):
#         """Forward the LeNet."""
#         conv_out = self.encoder(input)
#         feat = self.fc1(conv_out.view(-1, 50 * 4 * 4))
#         return feat
#
#
# class LeNetClassifier(nn.Module):
#     """LeNet classifier model for ADDA."""
#
#     def __init__(self):
#         """Init LeNet encoder."""
#         super(LeNetClassifier, self).__init__()
#         self.fc2 = nn.Linear(500, 10)
#
#     def forward(self, feat):
#         """Forward the LeNet classifier."""
#         out = F.dropout(F.relu(feat), training=self.training)
#         out = self.fc2(out)
#         return out



# def data_loader():
#     return img_src, img_tgt, label_src, label_tgt





