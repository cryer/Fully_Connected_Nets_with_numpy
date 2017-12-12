# Fully_Connected_Nets_with_numpy
Only use numpy to build FC Nets without any deep learning frame

不借助任何框架实现全连接神经网络，主体只用到了numpy。并实现一个监督二分类任务。

## 数据集准备
自己准备数据集，可以采用numpy的随机函数产生数据，我这里采用sklearn自带的数据集产生器,因为是分类任务，所以数据集做成了如下图的样子：

![](https://github.com/cryer/Fully_Connected_Nets_with_numpy/raw/master/image/1.png)

一共产生了400个样本数据，每个样本数据2个维度。

## 编写全连接层网络

全连接层网络采用模块化编程，方便移植和灵活的搭建不同层次的网络架构，这里利用编写好的模块搭建了三层的全连接神经网络，
大致架构为全连接--Relu--全连接--Relu--输出，损失采用softmax损失，方便应对多分类问题。softmax损失函数也是用numpy编写。
采用随机梯度下降进行优化。
## 效果测试
### 隐藏层单元为10-10时
  准确率71.25%,损失函数变化如下图：

![](https://github.com/cryer/Fully_Connected_Nets_with_numpy/raw/master/image/2.png)  

由于主要不是为了提升准确率，只是为了展示脱离框架编写全连接神经网络，因此不做再多优化。需要的可以自行优化，
比如增加结构的深度，当然这个例子比较简单，深了可能反而不好，也可以加入学习率退火，改进优化策略，用Adam之类的。


