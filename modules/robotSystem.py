import enum

class FunctionType(enum.Enum):
    '''系统工作模式'''
    autoaim = 1
    smallEnergy = 2
    bigEnergy = 3

class Robot():
    bulletSpeed = 10.0 # m/s
    delayTime = 20.0 # 系统延迟时间, ms
    def __init__(self) -> None:
        pass
