import time
import queue
import ctypes
from multiprocessing import Queue, RawArray, Lock
import modules.mvsdk as mvsdk
from modules.mindvision import Camera
from modules.communication import Communicator


class Sensor(Camera, Communicator):
    def __init__(self, exposure_ms: float, port: str, rx_queue: Queue, buffer: RawArray, lock: Lock):
        Camera.__init__(self, exposure_ms, True)
        Communicator.__init__(self, port)

        self.rx_queue = rx_queue
        self.buffer_addr = ctypes.addressof(buffer)
        self.lock = lock
        mvsdk.CameraSetCallbackFunction(self.camera, self.callback)

    @mvsdk.method(mvsdk.CAMERA_SNAP_PROC)
    def callback(self, hCamera, pRawData, pFrameHead, pContext):
        callback_time = time.time()

        success, message = self.receive_no_wait(True)
        if not success:
            mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)
            return

        head = pFrameHead[0]
        camera_timestamp = head.uiTimeStamp

        with self.lock:
            mvsdk.CameraImageProcess(hCamera, pRawData, self.buffer_addr, head)
        mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)

        try:
            self.rx_queue.put_nowait((callback_time, camera_timestamp, message))
        except queue.Full:
            print('Full!')
