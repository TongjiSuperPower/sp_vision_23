# 环境
[工业相机SDK](https://mindvision.com.cn/rjxz/list_12.aspx?lcid=138)

python 3.10，所需库见：requirements.py（未包含tensorflow）

# 说明
流程：识别 -> 分类 -> 预测 -> 通信

分别在modules文件夹下的detection.py、classification.py、prediction.py、communication.py中实现

mindvision.py和mvsdk.py中实现了工业相机的接口

utilities.py是一些用来调试看结果的函数

# Demo
main.py 实现了装甲板的识别、欧拉角的解算、与下位机的通信，未使用分类器和预测器

![img](assets/result.gif)