import cv2
from enum import IntEnum
from modules.io.parallel_camera import ParallelCamera
from modules.io.parallel_communicator import ParallelCommunicator
from modules.io.context_manager import ContextManager

class WorkMode(IntEnum):
    AUTOAIM = 0
    NASHOR = 1

def limit_degree(angle_degree: float) -> float:
    '''(-180,180]'''
    while angle_degree <= -180:
        angle_degree += 360
    while angle_degree > 180:
        angle_degree -= 360
    return angle_degree


def interpolate_degree(from_degree: float, to_degree: float, k: float) -> float:
    delta_degree = limit_degree(to_degree - from_degree)
    result_degree = limit_degree(k * delta_degree + from_degree)
    return result_degree


class Robot(ContextManager):
    def __init__(self, exposure_ms: float, port: str) -> None:
        self._camera = ParallelCamera(exposure_ms)
        self._communicator = ParallelCommunicator(port)

        self.send = self._communicator.send

        self.img: cv2.Mat = None
        self.img_time_s: float = None
        self.bullet_speed: float = None
        self.flag: int = None
        self.color: str = None
        self.id: int = None
        self.work_mode = WorkMode.AUTOAIM

    def _close(self) -> None:
        self._camera._close()
        self._communicator._close()
        print('Robot closed.')

    def update(self):
        '''注意阻塞'''
        self._camera.update()
        self.img = self._camera.img
        self.img_time_s = self._camera.read_time_s

        self._communicator.update()
        _, _, _, bullet_speed, flag = self._communicator.latest_status
        self.bullet_speed = bullet_speed if bullet_speed > 5 else 15

        # flag:
        # 个位: 1:英雄 2:工程 3/4/5:步兵 6:无人机 7:哨兵 8:飞镖 9:雷达站
        # 十位: TODO 用来切换自瞄/能量机关 1:自瞄 2:能量机关
        # 百位: 0:我方为红方 1:我方为蓝方
        self.flag = flag
        self.color = 'red' if self.flag < 100 else 'blue'
        self.id = self.flag % 100
        self.work_mode = WorkMode.AUTOAIM if (self.flag/10)%10 == 1 else WorkMode.NASHOR

    def yaw_pitch_degree_at(self, time_s: float) -> tuple[float, float]:
        '''注意阻塞'''
        while self._communicator.latest_read_time_s < time_s:
            self._communicator.update()

        for read_time_s, status in reversed(self._communicator.history):
            if read_time_s < time_s:
                time_s_before, status_before = read_time_s, status
                break
            time_s_after, status_after = read_time_s, status

        _, yaw_degree_after, pitch_degree_after, _, _, = status_after
        _, yaw_degree_before, pitch_degree_before, _, _, = status_before

        k = (time_s - time_s_before) / (time_s_after - time_s_before)
        yaw_degree = interpolate_degree(yaw_degree_before, yaw_degree_after, k)
        pitch_degree = interpolate_degree(pitch_degree_before, pitch_degree_after, k)

        return yaw_degree, pitch_degree
