import cv2
import time
import numpy as np
import modules.io.mvsdk as mvsdk
from modules.io.context_manager import ContextManager


class Camera(ContextManager):
    def __init__(self, exposure_ms: float) -> None:
        self._exposure_ms = exposure_ms
        self.read_time_s: float = None
        self._open()

    def _open(self) -> None:
        devices = mvsdk.CameraEnumerateDevice()
        if len(devices) < 1:
            raise RuntimeError("找不到相机设备")

        self._handle = mvsdk.CameraInit(devices[0])

        mvsdk.CameraSetAeState(self._handle, mvsdk.FALSE)  # 手动曝光
        mvsdk.CameraSetExposureTime(self._handle, self._exposure_ms * 1000)
        mvsdk.CameraSetFrameSpeed(self._handle, mvsdk.FRAME_SPEED_LOW)
        mvsdk.CameraSetIspOutFormat(self._handle, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

        buffer_size = 1024 * 1280 * 3
        self._buffer = mvsdk.CameraAlignMalloc(buffer_size)

        mvsdk.CameraPlay(self._handle)
        print('Camera opened.')

    def _close(self) -> None:
        mvsdk.CameraUnInit(self._handle)
        mvsdk.CameraAlignFree(self._buffer)
        print('Camera closed.')

    def read(self, buffer_address=None) -> tuple[bool, cv2.Mat | None]:
        try:
            raw, head = mvsdk.CameraGetImageBuffer(self._handle, 200)
            self.read_time_s = time.time()

            if buffer_address != None:
                mvsdk.CameraImageProcess(self._handle, raw, buffer_address, head)
                mvsdk.CameraReleaseImageBuffer(self._handle, raw)
                return True, None

            else:
                mvsdk.CameraImageProcess(self._handle, raw, self._buffer, head)
                mvsdk.CameraReleaseImageBuffer(self._handle, raw)
                img = (mvsdk.c_ubyte * head.uBytes).from_address(self._buffer)
                img = np.frombuffer(img, dtype=np.uint8)
                img = img.reshape((head.iHeight, head.iWidth, 3))
                return True, img

        except mvsdk.CameraException:
            return False, None

    def release(self) -> None:
        self._close()

    def reopen(self) -> None:
        '''注意阻塞'''
        self._close()

        last_error = None
        while True:
            try:
                self._open()
                break
            except Exception as error:
                if type(last_error) == type(error):
                    continue
                print(f'{error}')
                last_error = error

        print('Camera reopened.')
