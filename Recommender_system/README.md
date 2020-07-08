# 推荐系统
## 基础算法
基于物品的协同过滤： *itemCF.py*  
隐语义模型： *LFM.py*  
所有算法都在movieLens数据集上测试，结果如下：   

| algorithm | precision | recall | coverage |
| :-------: | :-------: | :----: | :------: |
|  itemCF   |  0.1888   | 0.2280 |  0.4078  |
|    LFM    |  0.0849   | 0.1051 |  0.1513  |


## 大文件
data/ml-1m文件夹下有多个大文件，使用Git lfs无效后上传至百度网盘  
* itemCF.py中的co_matrix.csv、item_sim.dict, [戳这里](https://pan.baidu.com/s/1EfFRFIGLStFyOgNa50VK8Q)，提取码：7l39  
* LFM.py训练好的lfm.model、R_dict.dict, [戳这里](https://pan.baidu.com/s/1RxU0iBf8P86j5NEC4idfog)，提取码：9wl2  