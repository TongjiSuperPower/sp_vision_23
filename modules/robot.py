import cv2
import time
import ctypes
import numpy as np
from multiprocessing import Process, Lock, Queue, RawArray
from modules.sensor import Sensor
from modules.communication import Communicator


def receiving(exposure_ms: float, port: str, rx_queue, buffer: RawArray, lock: Lock):
    Sensor(exposure_ms, port, rx_queue, buffer, lock)

    while True:
        time.sleep(1)


class Robot(Communicator):
    def __init__(self, exposure_ms: float, port: str) -> None:
        self.lock = Lock()
        self.buffer = RawArray(ctypes.c_uint8, 3932160)
        self.rx_queue = Queue(maxsize=1)
        self.rx_process = Process(target=receiving, args=(exposure_ms, port, self.rx_queue, self.buffer, self.lock))
        self.rx_process.start()

        self.callback_time: float = None
        self.camera_stamp: int = None
        self.serial_stamp: int = None
        self.frame: cv2.Mat = None
        self.yaw: float = None
        self.pitch: float = None
        self.bullet_speed: float = None
        self.flag: int = None

        self.rx_queue.get()
        Communicator.__init__(self, port)
        print('Robot initiated')

    def update(self):
        callback_time, camera_timestamp, message = self.rx_queue.get()

        with self.lock:
            img = np.frombuffer(self.buffer, dtype=np.uint8)
        img = img.reshape((1024, 1280, 3))

        self.callback_time = callback_time
        self.camera_stamp = camera_timestamp
        self.frame = img

        serial_stamp, yaw, pitch, bullet_speed, flag = message
        self.serial_stamp = serial_stamp
        self.yaw = yaw
        self.pitch = pitch
        self.bullet_speed = bullet_speed
        self.flag = flag
