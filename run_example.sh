#!/bin/bash
python -u alexnet_prune.py --epoch 140 --pretrained --prune 90 --logfolder 90 --resume 90/checkpoint.pth.tar /mnt/data/imagenet/ |& tee -a 90/run_3.log
