# SNN-Supervised-Learning
Spiking Neural Network for Supervised Learning using PyTorch

代码修改自https://github.com/yjwu17/BP-for-SpikingNN

使用脉冲神经网络实现MNIST、CIFAR10图像分类

### TODO

**Population coding**  ：在输出层使用N*M个神经元进行频率编码（rate coding），N是样本类别，对于MNIST和CIFAR10，N=10，M是种群神经元数量。每个label由M个神经元的总发放频率来确定。例如，若时间窗长度T=10，则一个神经元可以编码11种不同信息，若计算M个神经元的总的发放次数，则该population可以编码 **M\*(T+1)** 种信息，提高了神经元的表征能力。