# import os
# from os.path import join
# import numpy as np
# from PIL import Image
#
# import torch
# import torch.utils.data as data
# import torch.nn as nn
#
#
# # from torchvision import transforms
#
# def get_transform_dataset(dataset_name, rootdir, net_transform, downscale):
#     # user_paths = os.environ['PYTHONPATH'].split(os.pathsep)
#     transform, target_transform = get_transform2(dataset_name, net_transform, downscale)
#     return get_fcn_dataset(dataset_name, rootdir, transform=transform,
#                            target_transform=target_transform)
#
#
# def get_transform2(dataset_name, net_transform, downscale):
#     "Returns image and label transform to downscale, crop and prepare for net."
#     orig_size = get_orig_size(dataset_name)
#     transform = []
#     target_transform = []
#     if downscale is not None:
#         transform.append(transforms.Resize(orig_size // downscale))
#         target_transform.append(
#             transforms.Resize(orig_size // downscale,
#                               interpolation=Image.NEAREST))
#     transform.extend([transforms.Resize(orig_size), net_transform])
#     target_transform.extend([transforms.Resize(orig_size, interpolation=Image.NEAREST),
#                              to_tensor_raw])
#     transform = transforms.Compose(transform)
#     target_transform = transforms.Compose(target_transform)
#     return transform, target_transform
#
#
# def get_fcn_dataset(name, rootdir, **kwargs):
#     return dataset_obj[name](rootdir, **kwargs)
#
#
# dataset_obj = {}
#
#
# def register_dataset_obj(name):
#     def decorator(cls):
#         dataset_obj[name] = cls
#         return cls
#
#     return decorator
