# -*- coding: utf-8 -*-
"""
Created on 2021年1月26日21:37
@author: Haoyu Huang
Python 3.6.2
"""

from __future__ import print_function
import torchvision
import torchvision.transforms as transforms
import os
import time
from snn import *
import argparse

parser = argparse.ArgumentParser(description='Spiking Neural Network for Supervised Learning')
parser.add_argument('--epochs', type=int, default=100, help='Training epoch')
parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--device', type=int, default=-1, help='-1 for cpu training, 1,2,3,...for GPU device ID')
parser.add_argument('--model', type=str, default='CNN' ,help='MLP for Spike-MLP model or CNN for Spike-CNN model')
parser.add_argument('--optim', type=str, default='SGD', help='optimizer')
parser.add_argument('--lr', type=float, default=0.15, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD,default 0.5')
parser.add_argument('--ckpt_name', type=str, help='name of checkpoint file')
parser.add_argument('--dataset', type=str, default='CIFAR10',help='dataset')
parser.add_argument('--time_window', type=int, default=8,help='time window length for spike firing')
parser.add_argument('--scheduler', type=str, default='cos',help='learning rate scheduler, default for cosine')
parser.add_argument('--hiddens', type=int, default=128,help='number of hidden layer units for Spike-MLP')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
if args.device == -1:
    device = torch.device("cpu")
if args.device >= 0:
    device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else "cpu")

names = args.ckpt_name

# 特征图尺寸
cfg_kernel = [32, 16, 8, 4] # for CIFAR10 
# cfg_kernel = [28, 14, 7, 3] # for MNIST

# 选择数据集
data_path = './data/' #todo: input your data path
if args.dataset == 'MNIST':
    train_dataset = torchvision.datasets.MNIST(root= data_path, train=True, download=True, transform=transforms.ToTensor())
    test_set = torchvision.datasets.MNIST(root= data_path, train=False, download=True,  transform=transforms.ToTensor())
if args.dataset == 'CIFAR10':
    train_dataset = torchvision.datasets.CIFAR10(root= data_path, train=True, download=True,  transform=transforms.ToTensor())
    test_set = torchvision.datasets.CIFAR10(root= data_path, train=False, download=True,  transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

# choose SNN model
if args.dataset == 'MNIST' and args.model == 'CNN':
    snn = SCNN(input_channel=1,output_size=10,batch_size=args.batch_size,time_window=args.time_window,device=device).to(device)
if args.dataset == 'MNIST' and args.model == 'MLP':
    snn = SNN(input_size=28*28,output_size=10,hidden_units=args.hiddens,wins=args.time_window,batch_size=args.batch_size,device=device).to(device)
if args.dataset == 'CIFAR10' and args.model == 'CNN':
    snn = SCNN(input_channel=3,output_size=10,batch_size=args.batch_size,time_window=args.time_window,device=device).to(device)
if args.dataset == 'CIFAR10' and args.model == 'MLP':
    snn = SNN(input_size=32*32,output_size=10,hidden_units=args.hiddens,wins=args.time_window,batch_size=args.batch_size,device=device).to(device)

criterion = nn.MSELoss() # better than CrossEntropy in 10-class classification
if args.optim == 'Adam':
    optimizer = torch.optim.Adam(snn.parameters(), lr=args.lr)
if args.optim == 'SGD':
    optimizer = torch.optim.SGD(snn.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-5)
# 手动调整学习率
def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=40):
    """
    Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs
    """
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return optimizer
# 自动调整学习率
CosineLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0) # 余弦退火, T_max是cos周期1/4
MStepLR = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 300, 320, 340, 400], gamma=0.8) # gamma衰减乘数因子
StepLR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.65) # 固定步长衰减, gamma衰减因子
ExpLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98) # gamma指数衰减的底数

# 开始训练
acc_record = list([])
loss_train_record = list([])
loss_test_record = list([])
for epoch in range(args.epochs):
    running_loss = 0 # total loss in every 200 batches
    start_time = time.time() # 开始时间
    for i, (images, labels) in enumerate(train_loader):
        snn.zero_grad()
        optimizer.zero_grad()
        images = images.float().to(device)
        outputs = snn(images) # F.log_softmax(snn(images),-1)
        labels_ = torch.zeros(args.batch_size, 10).scatter_(1, labels.view(-1, 1), 1) # 将实数标签转化为one-hot向量
        # loss = -(outputs*labels_.to(device)).mean() # CrossEntropy Loss
        loss = criterion(outputs, labels_.to(device)) # MSE Loss
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        if (i+1)%200 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.5f'%(epoch+1, args.epochs, i+1, len(train_dataset)//args.batch_size,running_loss))
            running_loss = 0
    print('Time elasped:', time.time()-start_time)
    # testing 
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = snn(inputs) # F.log_softmax(snn(inputs),-1)
            labels_ = torch.zeros(args.batch_size, 10).scatter_(1, targets.view(-1, 1), 1)
            # loss = -(outputs.cpu()*labels_).sum()
            loss = criterion(outputs.cpu(), labels_)
            _, predicted = outputs.cpu().max(1)
            
            total += float(targets.size(0))
            correct += float(predicted.eq(targets).sum().item())
            if batch_idx % int(10000/args.batch_size) == 0:
                acc = 100. * float(correct) / float(total)
                # print(batch_idx, len(test_loader),' Acc: %.5f' % acc)
    print('Epoch: %d,Testing acc:%.3f'%(epoch+1,100*correct/total))
    acc = 100. * float(correct) / float(total)
    acc_record.append(acc)
    # 调整学习率
    if args.scheduler == 'cos':
        CosineLR.step()
    # 其余scheduler选项待添加
    # optimizer = lr_scheduler(optimizer, epoch, args.lr, 30) # 手动

    # model saving
    if epoch % 5 == 0:
        print('Saving..')
        state = {
            'net': snn.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'acc_record': acc_record,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_' + args.ckpt_name + '.t7')

