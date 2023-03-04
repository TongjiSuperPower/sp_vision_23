# RM对抗赛 - 能量机关识别与追踪视觉程序

## Identify & Tracking Baron Nahsor

### 原理
####  1.识别：

  \- 图像预处理（二值化、形态学运算） 

  \- 轮廓查找 

  \- 根据条件限制（限制父轮廓外接矩形的长宽比 & 轮廓占外接矩形的比例 & 目标装甲板占外接矩形的比例）找出装甲板

####  2. 定位：（旋转圆心的确定）

  \-  _拟合圆（在枪口晃动时不稳定，弃）_ 

  \- 根据条件限制寻找圆心R标志（限制R标轮廓长宽比 & 其到装甲板的距离比例 & R与父轮廓中心的连线角度）

####  3. 预测：

  \- 判断转向：定时提取多帧图像进行对比

  \- 距离检测：单目测距--pnp

  \- 转速计算：每隔0.2秒（刚好大于大符转动函数的一个周期），对比上次测量的装甲板中心与本次中心相较于转动圆心的角度差异，计算得到转速

  \- 大小符判断依据：比赛时间

​    \- 小符：匀速转动 10RPM，

​    \- 大符：存储一个前70次测得的转速数组n，对n进行小波去噪，并对其进行拟合，算出当前的相位，后续的速度计算均以拟合的函数为准。当检测的速度与拟合出函数所计算出的转速差别很大时，重新拟

####  4. 转角解算：

根据预测的ROI，使用斜抛模型和PnP，解出pitch轴和yaw轴转角，发给下位机



### 使用

- 开启大符模式时：

​    

```python
# 创建一个NahsorMarker类，参数为目标颜色（B：蓝色，R：红色），初始速度（m/s），是否调试，获取圆心的方法（默认为0，即通过几何特征筛选R标）
w = NahsorMarker('B', init_v=20, debug=0, get_R_method=0)
```



- 在使用循环内：

​    

```python
w.mark(frame, time_left=0) # 传入一帧图片，mark函数会完成 识别、定位、预测，无返回值；time_left为比赛剩余时间，用于大小符的判断
frame = w.markFrame() # 画图函数，返回标记好的一帧图像
res = w.getResult() # 获取需要的参数：状态（是否发现击打目标）、image_points、model_points
# 调用BulletModel进行转角解算
```



### 例程：

   ```python
from Nahsor import *
from module.ModuleCamera import Camera

def recognise(color, ns: NahsorSerial):
    temp = Camera(in_nums=0)
    temp.getf()

    # 新建能量机关对象
    w = NahsorMarker(color,  init_v=20, debug=0, get_R_method=0)   # 传入参数为 1.颜色代码：B/b -> 蓝色,  R/r -> 红色;

    while (cv2.waitKey(1) & 0xFF) != ord('q'):
        img = temp.getp()

        # 使用mark()方法，传入一帧图像
        w.mark(img, time_left=0)

        # 使用markFrame()获得标记好的输出图像
        img = w.markFrame()

        # 使用getResult()方法获得输出
        # print(w.getResult())

        cv2.imshow("Press q to end", img)
        temp.coff()


   ```