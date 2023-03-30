import cv2
import time
import queue
import ctypes
import numpy as np
from multiprocessing import Process, Queue, RawArray
from modules.mindvision import CallbackCamera
from modules.communication import Communicator


def communicating(port: str, tx: Queue, rx: Queue) -> None:
    communicator = Communicator(port)
    success_count = 0

    while True:
        time.sleep(0.001)

        # 发送命令
        try:
            command = rx.get_nowait()
            assert type(command) == tuple
            communicator.send(*command)
        except queue.Empty:
            pass

        # 接收机器人的状态
        success, state = communicator.receive_no_wait(False)
        if not success:
            continue

        try:
            tx.put_nowait(state)
        except queue.Full:
            try:
                tx.get_nowait()
            except queue.Empty:
                pass
            finally:
                try:
                    tx.put_nowait(state)
                except queue.Full:
                    print(f'State Queue Full! Successed Count: {success_count}')
                    success_count = -1

        success_count += 1


def capturing(exposure_ms: float, tx: Queue, buffer: RawArray) -> None:
    CallbackCamera(exposure_ms, tx, buffer)

    while True:
        time.sleep(10)


class Robot:
    def __init__(self, exposure_ms: float, port: str) -> None:
        self.buffer = RawArray(ctypes.c_uint8, 3932160)
        self.camera_rx = Queue(maxsize=1)
        self.capturing = Process(target=capturing, args=(exposure_ms, self.camera_rx, self.buffer))

        self.communicator_rx = Queue(maxsize=1)
        self.communicator_tx = Queue(maxsize=1)
        self.communicating = Process(target=communicating, args=(port, self.communicator_rx, self.communicator_tx))

        self.capturing.start()
        self.camera_rx.get()
        self.communicating.start()

        self.callback_time_s: float = None
        self.camera_stamp_ms: int = None
        self.img: cv2.Mat = None

        self.state_stamp: int = None
        self.yaw: float = None
        self.pitch: float = None
        self.bullet_speed: float = None
        self.flag: int = None

    def update(self) -> None:
        callback_time_s, camera_stamp_ms = self.camera_rx.get()
        img = np.frombuffer(self.buffer, dtype=np.uint8)
        img = img.reshape((1024, 1280, 3))

        state = self.communicator_rx.get()

        self.callback_time_s = callback_time_s
        self.camera_stamp_ms = camera_stamp_ms
        self.img = img

        state_stamp, yaw, pitch, bullet_speed, flag = state
        self.state_stamp = state_stamp
        self.yaw = yaw
        self.pitch = pitch
        self.bullet_speed = bullet_speed
        self.flag = flag

    def send(
        self,
        x_in_imu: float = 0, y_in_imu: float = 0, z_in_imu: float = 0,
        vx_in_imu: float = 0, vy_in_imu: float = 0, vz_in_imu: float = 0,
        stamp: int = 0, flag: int = 0,
    ) -> None:
        command = (x_in_imu, y_in_imu, z_in_imu, vx_in_imu, vy_in_imu, vz_in_imu, stamp, flag)
        try:
            self.communicator_tx.put_nowait(command)
        except queue.Full:
            try:
                self.communicator_tx.get_nowait()
            except queue.Empty:
                pass
            finally:
                try:
                    self.communicator_tx.put_nowait(command)
                except queue.Full:
                    print(f'Command Queue Full!')
