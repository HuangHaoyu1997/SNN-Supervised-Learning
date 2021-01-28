# -*- coding: utf-8 -*-
"""
Created on 2021年1月26日21:37
@author: Haoyu Huang
Python 3.6.2
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# 自制激活函数
class ActFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)  # 将Tensor转变为Variable保存到ctx中
        return input.gt(0.).float()  # input比0大返回True的float，即1.0，否则0.0

    @staticmethod
    def backward(ctx, grad_output, lens=0.5):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()

        # 膜电位处在发放阈值的一个邻域内，就令导数=1，否则为零
        # 采用矩形窗逼近脉冲函数的导数最简单，还可选高斯窗、三角窗等
        temp = abs(input) < lens 
        return grad_input * temp.float()

act_fun = ActFun.apply # 使用apply方法对自己定义的激活函数取个别名

def mem_update(ops, inputs, spike, mem, thr=0.3, v_th=0., decay=0.1, activation=None):
    '''
    SNN的更新函数，当成GRU来理解

    ops：可以是Conv操作，或Linear操作
    inputs: 输入的0-1脉冲张量
    spike: 上一个时刻的输出脉冲
    mem: 膜电位，类似GRU里的隐状态
    thr: 脉冲发放阈值
    decay: 膜电位衰减因子，实现Leaky IF的必要机制
    activation: 脉冲激活函数

    LIF模型的更新公式
    u_t+1 = k * u_t + \sum_j{ W_j o(j) }
    '''
    state = ops(inputs)  # 电流 I=\sum_j{ W_j o(j) }
    
    # mem是膜电位，spike=1即上一状态发放了脉冲，则mem减去一个发放阈值thr
    # spike=0则上一时刻未发放脉冲，则膜电位继续累积
    mem = state + (mem - spike * thr) * decay # .clamp(min=0., max=1.)
    now_spike = act_fun(mem - thr)
    return mem, now_spike.float()


class SNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_units, device, wins=10, batch_size=100,v_th_scales=0.8):  # 中间两个值应该与hidden_unit相等
        '''
        3-Layer Spiking MLP
        '''
        super(SNN, self).__init__()
        self.wins = wins # 脉冲发放时间窗长度
        self.batch_size = batch_size

        self.input_size = input_size
        self.hidden_units = hidden_units
        self.output_size = output_size
        self.device = device

        self.fc1 = nn.Linear(self.input_size, self.hidden_units, bias=True)
        self.fc2 = nn.Linear(self.hidden_units, self.hidden_units, bias=True)
        self.fc3 = nn.Linear(self.hidden_units, self.output_size, bias=True)  # linear readout layers

        # Learnable threshold
        self.v_th1 = nn.Parameter(v_th_scales * torch.rand(self.hidden_units, device=self.device))  # tensor变parameter，可训练
        self.v_th2 = nn.Parameter(v_th_scales * torch.rand(self.hidden_units, device=self.device))
        self.v_th3 = nn.Parameter(v_th_scales * torch.rand(self.output_size, device=self.device)) # 实际上没用到

        # Learnable decay
        self.decay1 = nn.Parameter(torch.rand(self.hidden_units, device=self.device))  # tensor变parameter，可训练
        self.decay2 = nn.Parameter(torch.rand(self.hidden_units, device=self.device))
        self.decay3 = nn.Parameter(torch.rand(self.output_size, device=self.device)) # 实际上没用到

    def forward(self, x):
        # 初始化膜电位和脉冲发放状态等变量
        h1_mem = h1_spike = h1_sumspike = torch.zeros(self.batch_size, self.hidden_units, device=self.device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(self.batch_size, self.hidden_units, device=self.device)
        h3_mem = h3_spike = h3_sumspike = torch.zeros(self.batch_size, self.output_size, device=self.device) # 实际上没用到
        
        for step in range(self.wins):
            x = x > torch.rand(x.size(), device = self.device) # 脉冲发放概率正比于像素值大小
            h1_mem, h1_spike = mem_update(fc=self.fc1, inputs=x.to(self.device), spike=h1_spike, mem=h1_mem, thr=0.3, v_th=self.v_th1, decay=self.decay1)

            h2_mem, h2_spike = mem_update(fc=self.fc2, inputs=h1_spike, spike=h2_spike, mem=h2_mem, thr=0.3, v_th=self.v_th2, decay=self.decay2)

            h2_sumspike = h2_sumspike + h2_spike # 累计时间窗内的脉冲数

        outs = self.fc3(h2_sumspike/self.wins)  # readout layers
        # outs = F.softmax(outs, dim=1) # 一般不采用softmax
        return outs

class SCNN(nn.Module):
    def __init__(self,input_channel,output_size,batch_size,time_window,device):
        super(SCNN, self).__init__()
        
        self.input_channel = input_channel
        self.output_size = output_size
        self.batch_size = batch_size
        self.device = device
        self.time_window = time_window

        self.conv1 = nn.Conv2d(input_channel, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(4 * 4 * 128, 256) # 输入尺寸是将最后一层feature map拉平
        self.fc2 = nn.Linear(256, output_size)

    def forward(self, input):
        # 计算output feature map size = (n-k+2p)/s+1
        # 若数据集为MNIST,需手动调整
        c1_mem = c1_spike = torch.zeros(self.batch_size, 32, 32, 32, device=self.device) # 第一层卷积层output feature map size=[batch,channel,width,height]
        c2_mem = c2_spike = torch.zeros(self.batch_size, 64, 16, 16, device=self.device) # 第二层卷积层output feature map size=[batch,channel,width,height]
        c3_mem = c3_spike = torch.zeros(self.batch_size, 128, 8, 8, device=self.device) # 第三层卷积层output feature map size=[batch,channel,width,height]

        h1_mem = h1_spike = h1_sumspike = torch.zeros(self.batch_size, 256, device=self.device) # 第一层全连接网络的输出维度
        h2_mem = h2_spike = h2_sumspike = torch.zeros(self.batch_size, self.output_size, device=self.device) # 第二层全连接网络的输出维度

        for step in range(self.time_window): # 仿真时间窗长度，即脉冲发放最大次数
            x = input > torch.rand(input.size(), device=self.device) # 脉冲发放概率正比于像素值大小

            c1_mem, c1_spike = mem_update(ops=self.conv1, inputs=x.float(), mem=c1_mem, spike=c1_spike)
            x = F.avg_pool2d(c1_spike, 2)

            c2_mem, c2_spike = mem_update(ops=self.conv2, inputs=x, mem=c2_mem, spike=c2_spike)
            x = F.avg_pool2d(c2_spike, 2)
            
            c3_mem, c3_spike = mem_update(ops=self.conv3, inputs=x, mem=c3_mem, spike=c3_spike)
            x = F.avg_pool2d(c3_spike, 2)

            x = x.view(self.batch_size, -1)
            h1_mem, h1_spike = mem_update(ops=self.fc1, inputs=x, mem=h1_mem, spike=h1_spike)
            h1_sumspike += h1_spike
            h2_mem, h2_spike = mem_update(ops=self.fc2, inputs=h1_spike, mem=h2_mem,spike=h2_spike)
            h2_sumspike += h2_spike

        outputs = h2_sumspike / self.time_window # rate coding
        return outputs
