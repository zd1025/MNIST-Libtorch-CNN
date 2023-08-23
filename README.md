# MNIST-Pytorch-CNN
手写识别MNIST（Pytorch + CNN实现）

[TOC]



## 一、项目启动

```
项目文件树
```

<img src="https://common-1316603156.cos.ap-shanghai.myqcloud.com/public/image-20230502101523577.png" style="zoom:80%;" />

```
可能出现的报错
```

- **Could not build wheels for opencv-python which use PEP 517 and cannot be installed directly**

  - ```
    关于cv2安装报错问题解决方案（anaconda下）->->->
    前景：本人的python环境是3.7 因此会出现如下错误
    报错：Could not build wheels for opencv-python which use PEP 517 and cannot be installed directly
    原因：cv2存在于opencv-python库中 直接进行pip出现报错
    解决方案：  1、前往网站 https://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv 下载对应 python 版本的 whl 文件
               2、放到本地 python 文件下的Lib文件夹中
               3、打开 cmd 进入whl所在文件夹下的路径 输入命令行 pip install whl文件名
               
    ```

- **ImportError: numpy.core.multiarray failed to import**

  - ```
    关于 opencv-python 与 numpy 不兼容的解决方案->->->
    报错：ImportError: numpy.core.multiarray failed to import
    解决方案：  1、pip uninstall numpy 卸载旧版本 numpy
               2、pip install numpy 最新版本 numpy 已经能适配
    ```

**博文参考链接：**```https://www.jb51.net/article/171584.htm```



## 二、网络结构

```
网络整体结构图
```

==首先对网络进行滤波处理 随后将数据集投入网络中训练==

![https://secure2.wostatic.cn/static/86rG8KMnprCPTrxbapdHLj/image.png?auth_key=1682994406-sA5vhPYnNB8ZuL2dP3eptN-0-0347354fa81b2484b2142e302a9a38c7&file_size=63296](https://secure2.wostatic.cn/static/86rG8KMnprCPTrxbapdHLj/image.png?auth_key=1682994406-sA5vhPYnNB8ZuL2dP3eptN-0-0347354fa81b2484b2142e302a9a38c7&file_size=63296)

<img src="https://secure2.wostatic.cn/static/pS1jqDw1cr3Nxz1bhwi8KJ/1682765284884.png?auth_key=1682994357-ddVwXebuGqDjJxXCPjy2Qc-0-71c1c34c49be50d96965b44ce570f2f6&file_size=34113" alt="https://secure2.wostatic.cn/static/pS1jqDw1cr3Nxz1bhwi8KJ/1682765284884.png?auth_key=1682994357-ddVwXebuGqDjJxXCPjy2Qc-0-71c1c34c49be50d96965b44ce570f2f6&file_size=34113" style="zoom:50%;" />

```
layer 1
```

<img src="https://secure2.wostatic.cn/static/esEBiNCqjYtZTZK5gjzwLH/image.png?auth_key=1682994506-hW4BNA5oZD9g9fCg5RADoy-0-3333e5ac44310b59af80284b20d3f7cb&file_size=29235" alt="https://secure2.wostatic.cn/static/esEBiNCqjYtZTZK5gjzwLH/image.png?auth_key=1682994506-hW4BNA5oZD9g9fCg5RADoy-0-3333e5ac44310b59af80284b20d3f7cb&file_size=29235" style="zoom:50%;" />

```
layer 2
```

<img src="https://secure2.wostatic.cn/static/uVRLWFb9Tbny2buVMPDUdb/image.png?auth_key=1682994530-r7PM3YTSWQg383Fy2esXW4-0-e4ae7b2e67137f8b5f414923a509a1d6&file_size=36079" alt="https://secure2.wostatic.cn/static/uVRLWFb9Tbny2buVMPDUdb/image.png?auth_key=1682994530-r7PM3YTSWQg383Fy2esXW4-0-e4ae7b2e67137f8b5f414923a509a1d6&file_size=36079" style="zoom: 50%;" />

```
layer 3
```

<img src="https://secure2.wostatic.cn/static/xqrkYRCXZfpwCptWPyEShX/image.png?auth_key=1682994568-34FEwU5fMKFxYpLWG1sEKK-0-12ef9352fa109f5a4961a0925230b93b&file_size=30391" alt="https://secure2.wostatic.cn/static/xqrkYRCXZfpwCptWPyEShX/image.png?auth_key=1682994568-34FEwU5fMKFxYpLWG1sEKK-0-12ef9352fa109f5a4961a0925230b93b&file_size=30391" style="zoom:50%;" />

```
layer 4
```

<img src="https://secure2.wostatic.cn/static/g8czqogjybCwgXbS4nw5L9/image.png?auth_key=1682994592-wvkXsGxGQSRXrzvKvXLjmC-0-4ef945c05b89763eafd988251b98edcd&file_size=27269" alt="https://secure2.wostatic.cn/static/g8czqogjybCwgXbS4nw5L9/image.png?auth_key=1682994592-wvkXsGxGQSRXrzvKvXLjmC-0-4ef945c05b89763eafd988251b98edcd&file_size=27269" style="zoom:50%;" />

```
full connection
```

<img src="https://secure2.wostatic.cn/static/f9UJzJKGwk69X61GBYWYin/image.png?auth_key=1682994619-dgjUge4T6FMrLRHkXsURb4-0-9a092d7d046d8864cc3c3d278584c187&file_size=32543" alt="https://secure2.wostatic.cn/static/f9UJzJKGwk69X61GBYWYin/image.png?auth_key=1682994619-dgjUge4T6FMrLRHkXsURb4-0-9a092d7d046d8864cc3c3d278584c187&file_size=32543" style="zoom:50%;" />



## 三、结果展示

```
训练时的loss曲线
```

<img src="https://secure2.wostatic.cn/static/eNGxRuFacUAFxXee9CVP9U/image.png?auth_key=1682994658-4pyCR7xEHotRTJy9CgW1Hq-0-ffe1f84ca08ddfa5071efecf738a0cd3&file_size=19419" alt="https://secure2.wostatic.cn/static/eNGxRuFacUAFxXee9CVP9U/image.png?auth_key=1682994658-4pyCR7xEHotRTJy9CgW1Hq-0-ffe1f84ca08ddfa5071efecf738a0cd3&file_size=19419" style="zoom:50%;" />

```
测试曲线（测试Loss曲线 / 测试的正确率曲线）
```

<img src="https://secure2.wostatic.cn/static/6K86VNbxHnowTNkMEA6tVo/image.png?auth_key=1682994704-sgYQ3H4iUDfQATrKmvfsE9-0-620d525f8530a8bff8eb9a31c946501a&file_size=16119" alt="https://secure2.wostatic.cn/static/6K86VNbxHnowTNkMEA6tVo/image.png?auth_key=1682994704-sgYQ3H4iUDfQATrKmvfsE9-0-620d525f8530a8bff8eb9a31c946501a&file_size=16119" style="zoom:50%;" />

<img src="https://secure2.wostatic.cn/static/a1hgshLmu4YTBpyWnHQamT/image.png?auth_key=1682994725-ak6dGwvNdxd7BnC5j9ihKG-0-925f4cdcd0ad924e2c2ea26236cfcc5f&file_size=19651" alt="https://secure2.wostatic.cn/static/a1hgshLmu4YTBpyWnHQamT/image.png?auth_key=1682994725-ak6dGwvNdxd7BnC5j9ihKG-0-925f4cdcd0ad924e2c2ea26236cfcc5f&file_size=19651" style="zoom:50%;" />



## 四、附上笔记链接

MNIST手写识别

https://www.wolai.com/oEcXbkg7gRz9kiKedVnmW5
