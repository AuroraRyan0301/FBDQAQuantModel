# FBDQAQuantModel
QuantModel for 2023 FBDQA(Financial Big Data and Quantitative Analytics)

## 设计思想
ResNet50卷积网络提高信噪比，筛选位置特征
- Transform encoder自注意力机制，对关键信号局部放大，提取有用信息
- ResBlock 再用一个残差块对高维有效信号做放大
- 线性层三分类
## 数据处理
本想用tsfresh等工具筛选因子，后来发现数据处理报错，而且现在效果还行，就没进一步处理。
## 模型结构
具体见代码
## 训练流程
先训练ResNet50，然后迁移学习到ResTransformer, 直接训ResTransformer参数量太大训不动。
## 效果
公榜测试，precision、f_score、pnl、pnl_average均第一，其中pnl、pnl_average断崖领先，提交模型pnl_average均超过万分之4手续费要求，可以实际投入使用

