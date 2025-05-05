复现实验来自[A Multimodal Deep Learning Approach for Typhoon Track Forecast by Fusing CNN and Transformer Structures](https://ieeexplore.ieee.org/document/10166556)

# 1.输入

## 1.1 ERA5数据

使用的225hPa、500hPa、700hPa三个层的位势高度、径向风、纬向风，构造时间步长为5的样本，形状为$(B,T,9,H,W)$ 9代表三个压力层的三个通道。

## 1.2 CMA BST

同样的采用步长为5构造样本，经纬度，风速，压力作为特征得到$(B,T,4)$的二维数据，并将最后一个时间步作为标签。

根据Bst得到的台风中心的经纬度，提取ERA5在中心位置的$10°\times10°$的区域作为输入。

# 2.网络结构

![image-20250505210857245](https://picbed-1313037164.cos.ap-nanjing.myqcloud.com/image-20250505210857245.png)

# 3. Run

在`main.py`中将`parser.add_argument('--is_train', action='store_true', default=0)`改为$1$即是训练