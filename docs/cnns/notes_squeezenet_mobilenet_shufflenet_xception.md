# 纵览轻量化卷积神经网络:SqueezeNet,MobileNet,ShuffleNet,Xception
[web](https://www.jiqizhixin.com/articles/2018-01-08-6)

由于这四种轻量化模型仅是在卷积方式上做了改变,因此本文仅对轻量化模型的创新点进行详细描述,对实验以及实现的细节感兴趣的朋友,请到论文中详细阅读.

## SqueezeNet
SqueezeNet由伯克利&斯坦福的研究人员合作发表于ICLR-2017.本文的新意是squeeze,squeeze在SqueezeNet中表示一个squeeze层,该层采用`1x1`卷积核对上一层`feature map`进行卷积,主要目的是减少`feature map`的维数.

创新点:

- 采用不同于传统的卷积方式,提出`fire module`:`squeeze层+expand层`

![](notes_squeezenet_mobilenet_shufflenet_xception01.png)

创新点与inception系列的思想非常接近!首先squeeze层,就是`1x1`卷积,其卷积核数要少于上一层`feature map`数,这个操作从inception系列开始就有了,并美其名压缩.Expand层分别用`1x1`和`3x3`卷积.然后concat,这个操作在inception系列里面也有.

