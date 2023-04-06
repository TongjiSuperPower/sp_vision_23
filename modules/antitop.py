#==================================
#how to use:
#from modules.antitop import *
#topStateDeque = TopStateDeque()
#antiTop = True   #开启反小陀螺模式
#
#if antiTop:
#   timeStampUs = cap.getTimeStampUs() if useCamera else int(time.time() * 1e6)
#   topStateDeque.insertArmorsDetection2(armors,timeStampUs / 1e6)
#   topStateDeque.getTopState2()
#===================================

useFPS = True  #如果是视频，是否用fps。因为python自带的time()包括处理时间，所以考虑用fps表示时间 t = △frame / fps
fps = 60  #自定义

from modules.armor_detection import *
from collections import deque

#TODO 
#带装甲板数字识别的小陀螺识别
#目前都是只能识别单个装甲板
#或许需要给Armor类加个成员

#需要用到的全局变量
weight_threshold = 150 #识别时容忍出现的左右偏差，大于这个值说明另一边装甲板进来了
dropframe_threshold =8 #容忍连续未检测到框的帧数，超过这个值就说明无检测了
top_threshold = 5 #判断是否为小陀螺的阈值,至少要转几次(可能一次是一个半圈)
rot_threshold = 5 #判断是否在旋转的阈值，旋转帧超过这个阈值且检测框位置变化很大说明转过了一个半圈


#====================
#用于小陀螺识别，装甲板状态类；
#数据成员：装甲板元组以及对应当前的时间戳
#用来表示在某一时刻下，识别的装甲板情况
#====================
class ArmorState:
    def __init__(self,armors:tuple[Armor],time:int) -> None:
        self.armors = armors
        self.timeStamp = time

    def getArmors(self) -> tuple[Armor]:
        return self.armors

    def getTimeStamp(self) -> int:
        return self.timeStamp  #应该是int吧..

    def getArmorState(self) -> tuple:  #应该是这么写的吧，反正也应该用不到
        return (self.armors,self.time)   

    def __str__(self) -> str:
        return f'ArmorState:armors:",{self.armors},"time:",{self.timeStamp}'

    def __repr__(self) -> str:
        return self.__str__()
        

#==================
#用于识别小陀螺的专用模块
#记录装甲板识别的状态的队列
#数据成员：armorsDetectionDeque：一个队列，大小为10
#          top_threshold：识别阈值
#如果呈现1-2-1-2-1...就说明是小陀螺
#deque有个不好的地方就是不能按索引，导致遍历必须浅拷贝到一个列表里
#TODO 后续考虑优化？
#==================
class TopStateDeque:
    def __init__(self) -> None:    
        self.armorsDetectionDeque = deque(maxlen=10) #轮流放入1..2..1..2用的，但为了方便检测和节省空间，省略其中的1（因为1和2连着，没啥用）
        #== 以上为基本数据成员 下面是扩展的 可能会用到也可能不会用到 取决于用什么方法 ==#
        self.armorsDetectionDiffList = []  #队列转列表，这个列表存的是时间差
        self.isDoubleBox = False  #判断当前是否该放进两个检测框
        
        lightbar = Lightbar(1,1,(1,1),'red',1,1) #初始化lastArmor用的，不然和空气比了，虽然可以加个对None的判断，但是感觉每次都判断一下有点费劲
        lightbarPair = LightbarPair(lightbar,lightbar,1,1,1) #初始化用的
        armor = Armor(lightbarPair,1,'red',None) #初始化用的

        self.lastArmorState = ArmorState(( armor,armor ),-1)  #上一次检测到的
        self.nowArmorState = ArmorState(( armor,armor ),-1)  #这一次检测到的

        self.nowArmorState.getArmors()[0].in_imu = np.array([[0,0,0]])  #别管这个，这是初始化用的，没了他会报错

        self.dropframe = 0
        self.rot = 0

        self.armorsDetectionDeque.append(self.nowArmorState)
    
    def getRear(self) -> ArmorState:  #获得队列尾（也就是刚放的），deque是从右到左的
        if len(self.armorsDetectionDeque)==0:
            return ArmorState((0,0),-1)
        armorstate = self.armorsDetectionDeque.pop()  #先把队尾刚放的pop出来再放回去
        self.armorsDetectionDeque.append(armorstate)
        return armorstate

    #======================
    #第二种检测方式
    def insertArmorsDetection2(self,armors:tuple[Armor],time:int) -> None:  #鉴于目前无法准确检测到两个框同时出现，这个“2”用框大范围横向平移去判断
        
        armorsNum = len(armors) #识别到的装甲板个数 

        #如果没检测到：
        if armorsNum == 0:
            #如果掉帧系数已到阈值：
            if self.dropframe == dropframe_threshold : #到达阈值
                self.armorsDetectionDeque.clear() #清空队列
                self.dropframe = 0 #重置
                return
            else: #如果掉帧系数没到阈值：
                self.dropframe += 1
                return
        
        #以下是检测到装甲板的处理
        self.lastArmorState = self.nowArmorState #更新上一次和这次的检测状态
        self.nowArmorState = ArmorState(armors,time)

        now_center = self.nowArmorState.getArmors()[0].in_imu.T[0]  #多个检测框时 以最左边为基准 center形式：[x y z]
        last_center = self.lastArmorState.getArmors()[0].in_imu.T[0] #TODO 以当前队列表示的数字为基准
      
        #如果 两次检测的中心差 小于一个阈值，就说明没有转半圈,记旋转值+1,return
        if abs(now_center[0] - last_center[0]) < weight_threshold :
            self.rot += 1
            return 
        else:   # 如果大于这个阈值：
            if self.rot >= rot_threshold :   #到达阈值且旋转值到阈值，也就是在转的状态且转过半圈
                if len(self.armorsDetectionDeque) > 0:
                    timediff = self.nowArmorState.getTimeStamp() - self.getRear().getTimeStamp()
                    if useFPS:
                        timediff = timediff / fps
                    xdiff = abs(last_center[0]- self.getRear().getArmors()[0].in_imu.T[0][0])/1000 #转换成m
                    rad = (xdiff/0.26) * 180 / np.pi
                    speed = rad / timediff
                    print("centers:",last_center[0],self.getRear().getArmors()[0].in_imu.T[0][0])
                    print("timediff:",self.lastArmorState.getTimeStamp() - self.getRear().getTimeStamp())
                    print("speed is:",speed) 
                self.armorsDetectionDeque.append(self.nowArmorState)  #加入队列,加入的是now，也就是转到边缘之后第一次出现板子的位置
                self.rot = 0
            else:  #如果到达阈值但是旋转值没到阈值，就说明没转，置零返回
                self.rot = 0
                self.armorsDetectionDeque.clear() #清空队列


    # 第二种检测方式的判断函数
    def getTopState2(self) -> int or float:  #判断是否是小陀螺,如果否：返回-1 如果是：返回解算的速度 不然再算速度要拷贝两次
        if len(self.armorsDetectionDeque) < top_threshold:  #如果队列中元素个数小于阈值,还无法判断
            return -1
        print("toping,and deque:",self.armorsDetectionDeque)

    def getTopState2(self) -> int or float:  #第二种判断方法
        pass


    #======================================
    #=========== 未使用或其他方法 ===========
    #===这些方法要求能够检测到左右都有的情况==
    #====目前无法采用=========================
    # 最开始的第一种检测方式，对检测精度要求高,搭配第一种判断方式
    def insertArmorsDetection(self,armors:tuple[Armor],time:int) -> None:  #设置当前某id装甲板数量以及当前时间戳,只有1..2..1..2时会插入队列
                                                                           #由于1和2相连，只存2
        armorsNum = len(armors) #识别到的装甲板个数
        self.lastArmorState = self.nowArmorState #更新上一次和这次的检测状态
        self.nowArmorState = ArmorState(armors,time)
        
        # if len(self.armorsDetectionDeque) == 0 and armorsNum != 0:   #当检测到框且队列为空时，先插入
        #     self.armorsDetectionDeque.append(self.nowArmorState)
            
        if armorsNum == 2 :
            if self.isDoubleBox == True:   #当检测到装甲板数量为2且不是相同时刻领域内的状况时，插入
                self.armorsDetectionDeque.append(self.nowArmorState)
                self.isDoubleBox = False    

                print("\n==============")
                print("当前检测到的装甲板数量、队列存储个数、队尾装甲个数、队尾时间戳")
                print(armorsNum,len(self.armorsDetectionDeque),len(self.getRear().getArmors()),self.getRear().getTimeStamp())
                print("deque:",self.armorsDetectionDeque)
        
        else:     #其他情况（如检测到、没检测或检测到两个但是是同一个时间点），置False
            self.isDoubleBox = True 

     # 搭配第一种检测方式的判断函数
    def getTopState(self) -> int or float:  #判断是否是小陀螺,如果否：返回-1 如果是：返回解算的速度 不然再算速度要拷贝两次
        if len(self.armorsDetectionDeque) < top_threshold:  #如果队列中元素个数小于阈值,还无法判断
            return False 
        #转列表,这个列表存储的是时间差而不是时间
        self.armorsDetectionDiffList.clear()
        _deque = self.armorsDetectionDeque.copy() #浅拷贝
        right = _deque.pop() #时间差的被减数，较新放的  队列中更靠右的 
        for i in range(top_threshold - 1):  #列表大小跟阈值有关,
            left = _deque.pop() #时间差的被减数，较新放的
            self.armorsDetectionDiffList.append(right.getTimeStamp() - left.getTimeStamp())  #减去较后放的
            right = left  #向前迭代
        print(self.armorsDetectionDiffList)
        var = np.var(self.armorsDetectionDiffList)
        print("方差为:",var)

    def insertArmorsDetection3(self,armors:tuple[Armor],time:int) -> None:  #设置当前某id装甲板数量以及当前时间戳,只有1..2..1..2时会插入队列
        armorsNum = len(armors) #识别到的装甲板个数
        self.lastArmorState = self.nowArmorState #更新上一次和这次的检测状态
        self.nowArmorState = ArmorState(armors,time)
       
        
        if len(self.armorsDetectionDeque) == 0 and armorsNum != 0:   #当检测到框且队列为空时，先插入
            self.armorsDetectionDeque.append(self.nowArmorState)
            
        if armorsNum != len(self.getRear().getArmors()) and armorsNum !=0 :
            self.armorsDetectionDeque.append(self.nowArmorState)
            
            print("\n==============")
            print("当前检测到的装甲板数量、队列存储个数、队尾装甲个数、队尾时间戳")
            print(armorsNum,len(self.armorsDetectionDeque),len(self.getRear().getArmors()),self.getRear().getTimeStamp())
            print("deque:",self.armorsDetectionDeque)

    def deque2list(self) -> None:  #用不到转换，白增加复杂度，因为转换是转全部，识别状态是按照阈值threshold转换的
        self.armorsDetectionDiffList.clear()
        _deque = self.armorsDetectionDeque.copy() #浅拷贝
        for i in range(len(self.armorsDetectionDeque)):
            self.armorsDetectionDiffList.append(_deque.pop())

    def getArmorsDetection(self) -> deque:
        return self.armorsDetectionDeque