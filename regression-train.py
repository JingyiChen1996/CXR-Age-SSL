import argparse
import os
import shutil
import time
import random

import numpy as np 
from tqdm import tqdm
from sklearn.metrics import r2_score

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F

# import model
from dnn121 import MobileNet
import wisenet as models
# import dataset
from regression_dataset import train_val_split, NIH_CXR_BASE, CxrDataset
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='MixMatch Training')
# Optimization options
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=32, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.002, type=float,
                    metavar='LR', help='initial learning rate')
# Checkpoints 
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Miscs
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
# Device options
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
# Method options
parser.add_argument('--error-range', type=int, default=9,
                        help='Number of labeled data')
parser.add_argument('--val-iteration', type=int, default=1024,
                        help='Validation iteration')
parser.add_argument('--out', default='result',
                        help='Directory to output the result')


args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    agrs.manualSeed = random.randint(1, 10000)
np.random.seed(args.manualSeed)

best_r2 = 0

def main():

    global best_r2

    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    # data
    # train-val split
    _ = train_val_split('~/cxr-jingyi/Age/NIH_train_10000.csv')
    train_set = CxrDataset(NIH_CXR_BASE, "~/cxr-jingyi/Age/train_8000.csv")
    val_set = CxrDataset(NIH_CXR_BASE, "~/cxr-jingyi/Age/val_2000.csv")
    test_set = CxrDataset(NIH_CXR_BASE, "~/cxr-jingyi/Age/NIH_test_2500.csv") 

    trainloader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # model
    print("==> creating model")

    def create_model(num_classes, ema=False):
        model = MobileNet(num_classes)
        #model = WideResNet(num_classes)
        model.cuda()

        if ema:
            for param in model.parameters():
                param.detach_()

        return model

    model = create_model(num_classes=1, ema=False)
    cudnn.benchmark = True
    print('Ttoal params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    start_epoch = 0

    # resume
    title = 'regression-NIH'
    if args.resume:
        # load checkpoints
        print('==> Resuming from checkpoints..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
        logger.set_names(['Train Loss','Valid Loss', 'Test Loss', \
                          'Train R2', 'Valid R2', 'Test R2', \
                          'Train Recall', 'Valid Recall', 'Test Recall'])

    writer = SummaryWriter(args.out)
    step = 0
    test_r2s = []
    # train and val
    for epoch in range(start_epoch, args.epochs):

        print('\nEpoch: [%d | %d] LR: %f' % (epoch+1, args.epochs, state['lr']))

        train_iteration, train_loss = train(trainloader, model, optimizer, criterion, epoch, use_cuda)
        train_acc_iteration, _, train_r2, train_recall = validate(trainloader, model, criterion, epoch, use_cuda, mode='Train Stats')
        val_iteration, val_loss, val_r2, val_recall = validate(val_loader, model, criterion, epoch, use_cuda, mode='Valid Stats')
        test_iteration, test_loss, test_r2, test_recall = validate(test_loader, model, criterion, epoch, use_cuda, mode='Test Stats')

        step = (epoch+1)

        writer.add_scalar('loss/train_loss', train_loss, step) 
        writer.add_scalar('loss/valid_loss', val_loss, step)
        writer.add_scalar('loss/test_loss', test_loss, step) 

        writer.add_scalar('R2/train_r2', train_r2, step)
        writer.add_scalar('R2/val_r2', val_r2, step)
        writer.add_scalar('R2/test_r2', test_r2, step)     

        writer.add_scalar('recall/train_recall', train_recall, step)
        writer.add_scalar('recall/val_recall', val_recall, step)
        writer.add_scalar('recall/test_recall', test_recall, step)                 
        
        # append logger file
        logger.append([train_loss, val_loss, test_loss, \
                       train_r2, val_r2, test_r2, \
                       train_recall, val_recall, test_recall])

        # save model
        is_best = val_r2 > best_r2
        best_r2 = max(val_r2, best_r2)
        save_checkpoint({
            'epoch': epoch+1,
            'state_dict': model.state_dict(),
            'acc': val_r2,
            'best_acc': best_r2,
            'optimizer': optimizer.state_dict()
        }, is_best)
        test_r2s.append(test_r2)
    logger.close()
    writer.close()

    print('Best R2:')
    print(best_r2)

    print('Mean R2:')
    print(np.mean(test_r2s[-20:]))


def train(train_loader, model, optimizer, criterion, epoch, use_cuda):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    bar = Bar('Training', max=args.val_iteration)
    t = tqdm(enumerate(train_loader), total=len(train_loader), desc='training')

    model.train()
    for batch_idx, (input, target) in t:
        if use_cuda:
            input, target = input.cuda(), target.cuda(non_blocking=True)
        # measure data loading time
        data_time.update(time.time()-end)
        # batch size
        batch_size = input.size(0)
        
        output = model(input)
        loss = criterion(output, target)

        # record loss
        losses.update(loss.item(), input.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time()-end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} '.format(
                    batch=batch_idx + 1,
                    size=args.val_iteration,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg
                    )
        bar.next()
    bar.finish()

    return (batch_idx, losses.avg,)


def validate(valloader, model, criterion, epoch, use_cuda, mode):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    r2s = AverageMeter()
    recalls = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar(f'{mode}', max=len(valloader))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            # measure data loading time
            data_time.update(time.time() - end)
            batch_size = inputs.size(0)
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            outputs_np = outputs.cpu().numpy()
            targets_np = targets.cpu().numpy()
            r2, recall = accuracy_reg(outputs_np, targets_np) 
            losses.update(loss.item(), inputs.size(0))
            r2s.update(r2, inputs.size(0))
            recalls.update(recall, inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | R2: {r2: .4f} | recall: {recall: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(valloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        r2=r2s.avg,
                        recall=recalls.avg,
                        )
            bar.next()
        bar.finish()
    return (batch_idx, losses.avg, r2s.avg, recalls.avg)

def save_checkpoint(state, is_best, checkpoint=args.out, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def linear_rampup(current, rampup_length=args.epochs):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

def accuracy_reg(output, target, error_range=args.error_range):

    """Computes the R2 score and recall for the specified age error range"""

    batch_size = len(target)
    r2 = r2_score(target, output) 

    correct = 0
    for target_, output_ in zip(target, output):
        print(target_[0], output_[0])
        if ((output_[0] >= (target_[0]-error_range)) and (output[0] <= (target_[0]+error_range))):
            correct += 1
    recall = correct/batch_size

    return r2, recall

if __name__ == '__main__':
    main()
