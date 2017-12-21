# Notes

## Data Science Bowl 2017
使用美国国家癌症研究所提供的数千个高分辨率肺部扫描数据集,参与者将开发算法,准确确定肺部病变何时发生癌变.这将大大减少困扰当前检测技术的假阳性率,使患者更早获得挽救生命的干预措施,并使放射科医师有更多时间与患者共同度过.

Access: https://www.kaggle.com/c/data-science-bowl-2017

- 冠军方案,solution-grt123-team,[code](https://github.com/lfz/DSB2017),[doc](https://github.com/lfz/DSB2017/blob/master/solution-grt123-team.pdf)
- 亚军方案,[summary of julian's](https://www.kaggle.com/c/data-science-bowl-2017/discussion/31551)

## Kaggle diabetic retinopathy
High-resolution retinal images that are annotated on a 0–4 severity scale by clinicians, for the detection of diabetic retinopathy. This data set is part of a completed Kaggle competition, which is generally a great source for publicly available data sets.

糖尿病视网膜病变:高分辨率视网膜图像由临床医生以0-4的严重程度进行标注,用于检测糖尿病视网膜病变.这个数据集是一个完整的Kaggle竞赛的一部分,这个竞赛通常是公开数据集的重要来源.

Access: https://www.kaggle.com/c/diabetic-retinopathy-detection

- 冠军方案,[Competition report](https://www.kaggle.com/c/diabetic-retinopathy-detection/discussion/15801)
- 亚军方案,[Team o_O Solution Summary](https://www.kaggle.com/c/diabetic-retinopathy-detection/discussion/15617)

## Cervical Cancer Screening
In this kaggle competition, you will develop algorithms to correctly classify cervix types based on cervical images. These different types of cervix in our data set are all considered normal (not cancerous), but since the transformation zones aren't always visible, some of the patients require further testing while some don't.

宫颈癌筛查:开发基于宫颈图像的正确分类子宫颈类型的算法.我们的数据集中的这些不同类型的子宫颈都被认为是正常的(不是癌变的),但是由于转化区并不总是可见的,一些患者需要进一步检测,而另一些则不需要.

Access: https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening/data

- 亚军方案,[Brief overview of #2 solution](https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening/discussion/35478)

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


---

# Challenges/Contest

## Grand Challenges in Biomedical Image Analysis
生物医学图像分析中的重大挑战

Access: https://grand-challenge.org/All_Challenges

### ENDOVIS
作为MICCAI的内镜视觉CAI挑战，我们的目标是提供一个评估当前艺术状态的正式框架，收集现场研究人员，并提供高质量的数据和验证内窥镜视觉算法的协议。 相关：ISBI 2017主办：grand-challenge.org

### CAMELYON17
在组织学淋巴结切片的全幻灯片图像中乳腺癌转移的自动检测和分类。这项任务具有很高的临床意义，通常需要病理学家进行广泛的微观评估。 与：MICCAI 2017相关主办：grand-challenge.org

### 润饰
视网膜OCT液体挑战的目标是比较能够检测和分割不同类型的视网膜流体病变的自动化算法，所述算法能够在来自不同制造商的设备获得的代表不同的视网膜疾病的光学相干断层扫描（OCT）体积的通用数据集上进行检测和分割。 打开提交数据下载主办：grand-challenge.org

### 白内障
cataRACT手术自动工具注解的挑战旨在评估世界上最常见手术过程中基于图像的工具检测算法。 打开提交数据下载主办：grand-challenge.org

### 蝌蚪
EuroPOND协会与阿尔茨海默病神经影像行动（ADNI）合作，为您提供阿尔茨海默病预测纵向进展（TADPOLE）的挑战。 相关：AAPM 2017主办：grand-challenge.org

### LCTSC
这个挑战将提供一个平台，用于比较各种自动/半自动分割算法，当它们被用于描绘用于胸部放射治疗计划的CT图像风险器官时。 相关：MVIP 2017主办：grand-challenge.org

### ROCC
视网膜OCT分类挑战赛（ROCC）与MVIP2017共同组织为一天的挑战赛。这个挑战的目标是调用不同的自动算法，能够在普通的OCT体数据集上检测来自正常视网膜的DR病，这些数据集是使用Topcon SD-OCT设备获取的。 相关：RSNA 2017

### 小儿骨龄挑战
开发一种算法，该算法能够使用超过12,000张图像的大型数据集，最精确地确定儿科手部X光片上验证集的骨龄。 与：MICCAI 2017相关

### 结直肠癌肝转移生存预测
该挑战的目的是使用来自Memorial Sloan Kettering癌症中心的数据，基于来自对比增强肝脏CT扫描和患者临床变量的预测因子预测肝脏无病存活。 与：MICCAI 2017相关

### 数字病理学：整体幻灯片分类
这一挑战的重点是将NSCLC患者分为腺癌和鳞状细胞癌，将HNSCC患者分为HPV +/-和K17（mRNA）+/-分子亚型。 与：MICCAI 2017相关

### 数字病理学：核分割
分割从非小细胞肺癌，头颈部鳞状细胞癌，多形性胶质母细胞瘤和低级别胶质瘤肿瘤患者获得的整个载玻片组织的细胞核 与：MICCAI 2017相关

### 缺血性中风病变分割2017年
ISLES 2017要求基于急性MRI数据预测卒中病变结果的方法。提供了48个脑卒中患者和匹配专家分割的多光谱数据集。 相关：AAPM 2017

### 前列腺2号挑战
预测包括轴向和矢状T2加权像，Ktrans图像（通过动态增强计算）和表观扩散系数（ADC）图像组成的前列腺MRI检查的格里森级。 与：MICCAI 2017相关

### 6个月的婴儿脑部MRI分割
iSeg-2017比较（半）自动算法的6个月婴儿脑组织分割和测量相应的结构使用T1和T2加权脑MRI扫描。 与：MICCAI 2017相关

### 冠状动脉重建的挑战
CoronARe比较了介入C型臂旋转血管造影冠状动脉三维重建准确性的最新方法。 与：MICCAI 2017相关

### WMH分割挑战
这个挑战直接比较了从FLAIR和T1加权MR图像自动分割假定血管起源的白质高强度的方法。 与：Kaggle相关

### 宫颈癌筛查
开发一种从图像中准确识别女性宫颈类型的算法。这样做可以防止无效的治疗，并帮助转诊需要更高级治疗的病例。 与：MICCAI 2017相关

### ENIGMA小脑
这个挑战调查小脑小叶分割和标签的进展，围绕三个MRI数据集。 关联于：ISMRM 2017

### 追踪
TraCED的目标是评估使用临床上可行的MR成像序列的常见和新兴纤维跟踪管线/算法的可重复性。 与DSB 2017相关

### 数据科学碗2017年
你能改善肺癌的检测吗？给予CT扫描，预测是否有肺癌。2017年数据科学碗有100万美元的奖金。 打开提交相关：ISBI 2017

### 皮肤病变分析对黑色素瘤的检测
从国际皮肤影像协会提供的dermoscopic图像分割，分析和诊断皮肤癌。 打开提交相关：ISBI 2017

### 甲状腺癌的组织芯片分析
组织微阵列（TMAs）可以提供新的生物标志物，可能对诊断，预测结果和对治疗的反应有价值。这个挑战的目标是建立TMA预测甲状腺癌的模型。 打开提交与：MICCAI 2017相关托管于：codalab.org

### LiTS - 肝肿瘤分割
我们鼓励研究人员开发自动分割算法，在对比增强的腹部CT扫描中分割肝脏病变。数据和分割由世界各地的临床网站提供。 打开提交相关：SPIE MI 2017

### PROSTATEx
使用定量图像分析方法对临床上显着的前列腺病变的诊断分类。

## Visual Concept Extraction Challenge in Radiology
Manually annotated radiological data of several anatomical structures (e.g. kidney, lung, bladder, etc.) from several different imaging modalities (e.g. CT and MR). They also provide a cloud computing instance that anyone can use to develop and evaluate models against benchmarks.

Access: http://www.visceral.eu/

## Grand Challenges in Biomedical Image Analysis
A collection of biomedical imaging challenges in order to facilitate better comparisons between new and existing solutions, by standardizing evaluation criteria. You can create your own challenge as well. As of this writing, there are 92 challenges that provide downloadable data sets.

Access: http://www.grand-challenge.org/

## Dream Challenges
DREAM Challenges pose fundamental questions about systems biology and translational medicine. Designed and run by a community of researchers from a variety of organizations, our challenges invite participants to propose solutions — fostering collaboration and building communities in the process. Expertise and institutional support are provided by Sage Bionetworks, along with the infrastructure to host challenges via their Synapse platform. Together, we share a vision allowing individuals and groups to collaborate openly so that the “wisdom of the crowd” provides the greatest impact on science and human health.

- The Digital Mammography DREAM Challenge.
- ICGC-TCGA DREAM Somatic Mutation Calling RNA Challenge (SMC-RNA)
- DREAM Idea Challenge
- These were the active challenges at the time of adding, many more past challenges and upcoming challenges are present!

Access: http://dreamchallenges.org/

## Multiple sclerosis lesion segmentation
challenge 2008. A collection of brain MRI scans to detect MS lesions.

Access: http://www.ia.unc.edu/MSseg/

## Multimodal Brain Tumor Segmentation Challenge
Large data set of brain tumor magnetic resonance scans. They’ve been extending this data set and challenge each year since 2012.

Access: http://braintumorsegmentation.org/

## Coding4Cancer
A new initiative by the Foundation for the National Institutes of Health and Sage Bionetworks to host a series of challenges to improve cancer screening. The first is for digital mammography readings. The second is for lung cancer detection. The challenges are not yet launched.

Access: http://coding4cancer.org/

## EEG Challenge Datasets on Kaggle
- Melbourne University AES/MathWorks/NIH Seizure Prediction, Predict seizures in long-term human intracranial EEG recordings. Access: https://www.kaggle.com/c/melbourne-university-seizure-prediction. 预测长期人类颅内脑电图记录中的癫痫发作
- 
American Epilepsy Society Seizure Prediction Challenge, Predict seizures in intracranial EEG recordings. Access: https://www.kaggle.com/c/seizure-prediction. 预测颅内脑电图记录中的癫痫发作
- UPenn and Mayo Clinic's Seizure Detection Challenge, Detect seizures in intracranial EEG recordings. Access: https://www.kaggle.com/c/seizure-detection. 检测颅内脑电图记录中的癫痫发作
- Grasp-and-Lift EEG Detection, Identify hand motions from EEG recordings. Access: https://www.kaggle.com/c/grasp-and-lift-eeg-detection. 识别来自脑电图记录的手部运动

## Challenges track in MICCAI Conference
The Medical Image Computing and Computer Assisted Intervention. Most of the challenges would've been covered by websites like grand-challenges etc. You can still see all of them under the "Satellite Events" tab of the conference sites.

- 2017 - http://www.miccai2017.org/satellite-events
- 2016 - http://www.miccai2016.org/en/SATELLITE-EVENTS.html
- 2015 - https://www.miccai2015.org/frontend/index.php?page_id=589

Access: http://www.miccai.org/ConferenceHistory

## International Symposium on Biomedical Imaging (ISBI)
The IEEE International Symposium on Biomedical Imaging (ISBI) is a scientific conference dedicated to mathematical, algorithmic, and computational aspects of biomedical imaging, across all scales of observation. Most of these challenges will be listed in grand-challenges. You can still access it by visiting the "Challenges" tab under "Program" in each year's website.

- 2017 - http://biomedicalimaging.org/2017/challenges/
- 2016 - http://biomedicalimaging.org/2016/?page_id=416

Access: http://biomedicalimaging.org
