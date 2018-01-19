# 图片标注工具

## labelme
- github: https://github.com/wkentaro/labelme

## LabelMeAnnotationTool
- github: https://github.com/CSAILVision/LabelMeAnnotationTool
- online: http://labelme.csail.mit.edu/
- online: http://labelme2.csail.mit.edu/Release3.0/index.php

### From Dockerfile

    $ sudo service docker restart
    $ sudo docker build -t hejian/labelme -f Dockerfile github.com/CSAILVision/LabelMeAnnotationTool
    $ sudo docker run --name labelme -d -p 9000:80 hejian/labelme

浏览器访问:`http://10.0.1.159:9000/`或`http://10.0.1.159:9000/tool.html`.此时使用自带的默认数据集`labelme`工作.

### 新增数据
上传数据到服务器:

    $ scp -qr mb1_base_may_80/ root@118.190.99.177:/root/hej/mb1_base_may_80_new

首先拷贝数据到容器中:

    $ sudo docker cp /root/hej/mb1_base_may_80_new/ labelme:/var/www/html/Images/mb1_base_may_80_new

然后访问:

http://118.190.99.177:9000/tool.html?tool.html?username=flystarhe&mode=f&folder=mb1_base_may_80_new&image=img1.jpg

拷贝结果到主机:

    $ sudo docker cp labelme:/var/www/html/Masks/mb1_base_may_80_new/ /root/hej/mb1_base_may_80_mask

下载数据到本地:

    $ scp -qr root@118.190.99.177:/root/hej/mb1_base_may_80_mask/ mb1_base_may_80_mask

### 图像集合
通过在命令行上运行以下命令来创建要标记的图像集合:

    $ cd ./annotationTools/sh/
    $ ./populate_dirlist.sh

这将创建"./Images"文件夹内所有图像的列表,并将出现在文件"./annotationCache/DirLists/labelme.txt"中.然后,可以使用以下URL在集合中标记图像:

http://118.190.99.177:9000/tool.html?tool.html?username=flystarhe&collection=labelme&mode=f&folder=example_folder

可以通过从命令行运行以下命令来创建包含特定文件夹的集合:

    $ cd ./annotationTools/sh/
    $ ./populate_dirlist.sh my_collection.txt example_folder

该列表将显示在"./annotationCache/DirLists/my_collection.txt"中.然后,可以使用以下URL在集合中标记图像:

http://118.190.99.177:9000/tool.html?tool.html?username=flystarhe&collection=my_collection&mode=f&folder=example_folder
