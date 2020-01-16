import argparse
import os
import shutil
import time
import random
import math
import pandas as pd
import numpy as np 
from tqdm import tqdm
import pdb
import bisect

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F

# import model
from dnn121 import DenseNet121, MobileNet
import wisenet as models
# import dataset
from mixmatch_dataset import train_val_split, NIH_CXR_BASE, CxrDataset, CXR_unlabeled
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from tensorboardX import SummaryWriter



test_set = CxrDataset(NIH_CXR_BASE, "~/cxr-jingyi/Age/NIH_test_2500.csv") 
test_loader = data.DataLoader(test_set, batch_size=32, shuffle=False, num_workers=32)

model = MobileNet(16)
model = model.cuda()
#checkpoint = torch.load('/home/jingyi/cxr-jingyi/Age/result/supervised/model_best.pth.tar')
checkpoint = torch.load('/home/jingyi/cxr-jingyi/Age/checkpoint/cifar10-semi/exp/ckpt.pth.tar')
#model.load_state_dict(checkpoint['state_dict'])
model.load_state_dict(checkpoint['net'])

def validate(val_loader, model, mode = 'valid'):
    
    top1 = AverageMeter()
    top5 = AverageMeter()
    predict = []

    # switch to evaluate mode
    model.eval()
    
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            sl = input.shape
            batch_size = sl[0]
            input = input.cuda()
            target = target.cuda(async=True)

            # compute output
            output = model(input)
            softmax = torch.nn.LogSoftmax(dim=1)(output)
            _, pred = output.topk(5, 1, True, True)
            #_, pred = torch.max(output, dim=1)
            predict.append(pred.data.cpu().data.numpy())
 
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

    print(' ****** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    
    return top1.avg, top5.avg, predict

prec1, prec5, predict = validate(test_loader, model, 'test')
print('prec1: {}, prec5: {}'.format(prec1, prec5))
#test_df = test_loader.dataset.entries

#out = []
#for i in range(len(predict)):
    #for j in range(len(predict[i])):
        #out.append(predict[i][j])
#print(len(out))

#test_df['predict'] = out
#test_df.to_csv('~/cxr-jingyi/Age/test_predict.csv', index=False)
