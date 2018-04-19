import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from alexnet_mod import alexnet
from pruning.methods import weight_prune
from pruning.utils import prune_rate

parser = argparse.ArgumentParser(
        description='PyTorch AlexNet: Check Pruning Rate')

parser.add_argument('weights', metavar='DIR',
                    help='path to pickled weights')

args = parser.parse_args()
model = alexnet(pretrained=True)
model.features = torch.nn.DataParallel(model.features)
model.cuda()

print("=> loading checkpoint '{}'".format(args.weights))
checkpoint = torch.load(args.weights)
params = {k: v for k, v in checkpoint['state_dict'].items() if 'mask' not in k}
mask_params = {k: v for k, v in checkpoint['state_dict'].items() if 'mask' in k}
model.load_state_dict(params)
model.set_masks(list(mask_params.values()))
prune_rate(model)
