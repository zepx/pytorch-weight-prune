"""
Pruning a MLP by weights with one shot
"""

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import argparse
import os

from pruning.methods import weight_prune
from pruning.utils import to_var, train, test, prune_rate
from models import MLP, AlexNet

parser = argparse.ArgumentParser(description='PyTorch Prune AlexNet and ReTraining')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--prune', default=90., type=float,
                    help='Pruning percentage')

args = parser.parse_args()
# Hyper Parameters
param = {
    'pruning_perc': args.prune,
    'batch_size': args.batch_size, 
    'test_batch_size': 100,
    'num_epochs': args.epochs,
    'learning_rate': args.lr,
    'weight_decay': args.weight_decay,
}


# Data loaders
# train_dataset = datasets.MNIST(root='../data/',train=True, download=True, 
    # transform=transforms.ToTensor())
# loader_train = torch.utils.data.DataLoader(train_dataset, 
    # batch_size=param['batch_size'], shuffle=True)

# test_dataset = datasets.MNIST(root='../data/', train=False, download=True, 
    # transform=transforms.ToTensor())
# loader_test = torch.utils.data.DataLoader(test_dataset, 
    # batch_size=param['test_batch_size'], shuffle=True)


# Data loading code for AlexNet
traindir = os.path.join(args.data, 'ILSVRC2012_img_train_pytorch')
valdir = os.path.join(args.data, 'ILSVRC2012_img_val_pytorch')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

train_sampler = None
loader_train = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
    num_workers=args.workers, pin_memory=True, sampler=train_sampler)

loader_test = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)

# Load the pretrained model
net = AlexNet()
net.load_state_dict(torch.load('/home/choong/.torch/models/alexnet-owt-4df8aa71.pth'), strict=False)
if torch.cuda.is_available():
    print('CUDA enabled.')
    net.cuda()
print("--- Pretrained network loaded ---")
# test(net, loader_test)

# prune the weights
masks = weight_prune(net, param['pruning_perc'])
net.set_masks(masks)
net = nn.DataParallel(net)
print("--- {}% parameters pruned ---".format(param['pruning_perc']))
test(net, loader_test)


# Retraining
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(net.parameters(), lr=param['learning_rate'], 
                                weight_decay=param['weight_decay'])

train(net, criterion, optimizer, param, loader_train)


# Check accuracy and nonzeros weights in each layer
print("--- After retraining ---")
test(net, loader_test)
prune_rate(net)


# Save and load the entire model
torch.save(net.state_dict(), 'models/alexnet_pruned.pkl')
