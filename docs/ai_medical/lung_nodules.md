## 肺部结节智能诊断
大赛要求参赛者使用患者的CT影像设计算法,训练模型,在独立的测试数据集中找出CT影像中的肺部结节的位置并给出是真正肺结节的概率.竞赛期望选手可以提出好的算法,以达到辅助医生进行肺结节诊断的目的.

数据由大赛合作医院授权提供,有数千份高危患者的低剂量肺部CT影像(mhd格式)数据,每个影像包含一系列胸腔的多个轴向切片.每个影像包含的切片数量会随着扫描机器/扫描层厚和患者的不同而有差异.所有CT影像数据严格按照国际通行的医疗信息脱敏标准进行脱敏处理,切实保障数据安全.

Access: https://tianchi.aliyun.com/competition/introduction.htm?raceId=231601

### 天池医疗AI大赛[第一季],Rank1解决方案
- https://tianchi.aliyun.com/competition/new_articleDetail.html?spm=5176.8366600.0.0.6b120f825NGOIR&raceId=231601&postsId=2947

### 天池医疗AI大赛[第一季] Rank2解决方案
- https://tianchi.aliyun.com/competition/new_articleDetail.html?spm=5176.8366600.0.0.6b120f825NGOIR&raceId=231601&postsId=2966

### 天池医疗AI大赛[第一季] Rank3解决方案
- https://tianchi.aliyun.com/competition/new_articleDetail.html?spm=5176.8366600.0.0.6b120f825NGOIR&raceId=231601&postsId=2893

### 天池医疗AI大赛[第一季] Rank4解决方案
- https://tianchi.aliyun.com/competition/new_articleDetail.html?spm=5176.8366600.0.0.6b120f825NGOIR&raceId=231601&postsId=2915

### 天池医疗AI大赛[第一季] Rank7解决方案
- https://tianchi.aliyun.com/competition/new_articleDetail.html?spm=5176.9876270.0.0.56a75983P3Xwzh&raceId=231601&postsId=2898
- [video](https://tianchi.aliyun.com/competition/videoStream.html#postsId=3489)
- [code](https://github.com/YiYuanIntelligent/3DFasterRCNN_LungNoduleDetector)

### 天池医疗AI大赛[第一季] Rank8解决方案
- https://tianchi.aliyun.com/competition/new_articleDetail.html?spm=5176.8366600.0.0.beafd11zG4odT&raceId=231601&postsId=3099
- [code](https://github.com/daichengasda/Tianchi)

1. 基于3D CNN+Unet/Inception/Resnet的分割网络,用以找出疑似结点.利用结节标注信息生成的结节mask图像,训练基于卷积神经网络的肺结节分割器
2. 基于ResNet的分类网络,判断每一个疑似结点是否是真阳性.找到疑似肺结节后,可以使用图像分类算法对疑似肺结节进行分类,得出疑似肺结节是否为真正肺结节的概率
3. 利用类似adaboost的方法训练多个一样的分类模型,不断的提升分类的准确率

Unet是一个在医学图像处理领域,应用很广泛的网络结构,是一种全卷积神经网络,输入和输出都是图像,没有全连接层.

## Others
- [LIDC&LUNA16数据说明](http://www.jianshu.com/p/e3e4984833dd?spm=5176.9876270.0.0.3cf4ab00AOROvg)
- [armamut/Getting the lungs right](https://www.kaggle.com/armamut/getting-the-lungs-right)

---


数据预处理—肺部区域提取:

![http://aliyuntianchipublic.cn-hangzhou.oss-pub.aliyun-inc.com/public/files/image/1095279155895/1507289164706_7FJAflCvF5.jpg]