import cv2
import time
import queue
import ctypes
import numpy as np
from multiprocessing import Queue, RawArray
import modules.mvsdk as mvsdk


class Camera:
    def __init__(self, exposure_ms: float = None, has_trigger: bool = False) -> None:
        # 枚举相机
        devices = mvsdk.CameraEnumerateDevice()
        if len(devices) < 1:
            raise RuntimeError("找不到相机设备")

        # 打开相机
        self.camera = mvsdk.CameraInit(devices[0])

        # 获取相机特性描述
        capability = mvsdk.CameraGetCapability(self.camera)
        self.isMono = capability.sIspCapacity.bMonoSensor == mvsdk.TRUE
        self.width = capability.sResolutionRange.iWidthMax
        self.height = capability.sResolutionRange.iHeightMax

        # 黑白相机让ISP直接输出MONO数据，而不是扩展成R=G=B的24位灰度
        if self.isMono:
            mvsdk.CameraSetIspOutFormat(self.camera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
        else:
            mvsdk.CameraSetIspOutFormat(self.camera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

        # 相机参数设置
        mvsdk.CameraSetTriggerMode(self.camera, 2 if has_trigger else 0)  # 0:连续采集 1:软触发 2:硬触发
        mvsdk.CameraSetFrameSpeed(self.camera, mvsdk.FRAME_SPEED_NORMAL)  # NORMAL帧率可以达到200fps
        if exposure_ms != None:
            mvsdk.CameraSetAeState(self.camera, mvsdk.FALSE)  # 手动曝光
            mvsdk.CameraSetExposureTime(self.camera, exposure_ms * 1000)  # 曝光时间ms

        # 让SDK内部取图线程开始工作
        mvsdk.CameraPlay(self.camera)

        # 分配buffer，用来存放ISP输出的图像
        bufferSize = self.width * self.height * (1 if self.isMono else 3)
        self.buffer = mvsdk.CameraAlignMalloc(bufferSize, 16)

    def read(self) -> tuple[bool, cv2.Mat | None]:
        try:
            rawData, head = mvsdk.CameraGetImageBuffer(self.camera, 200)
            mvsdk.CameraImageProcess(self.camera, rawData, self.buffer, head)
            mvsdk.CameraReleaseImageBuffer(self.camera, rawData)

            frameData = (mvsdk.c_ubyte * head.uBytes).from_address(self.buffer)
            frame = np.frombuffer(frameData, dtype=np.uint8)
            frame = frame.reshape((head.iHeight, head.iWidth, 1 if self.isMono else 3))
            return True, frame
        except mvsdk.CameraException:
            return False, None

    def get_stamp_ms(self) -> int:
        return mvsdk.CameraGetFrameTimeStamp(self.camera) / 1e3

    def release(self) -> None:
        # 关闭相机
        mvsdk.CameraUnInit(self.camera)
        # 释放帧缓存
        mvsdk.CameraAlignFree(self.buffer)


class CallbackCamera(Camera):
    def __init__(self, exposure_ms: float, tx: Queue, buffer: RawArray):
        Camera.__init__(self, exposure_ms, True)

        self.tx = tx
        self.buffer_address = ctypes.addressof(buffer)
        mvsdk.CameraSetCallbackFunction(self.camera, self.callback)

        self.success_count = 0

    @mvsdk.method(mvsdk.CAMERA_SNAP_PROC)
    def callback(self, hCamera, pRawData, pFrameHead, pContext):
        callback_time_s = time.time()

        head = pFrameHead[0]
        camera_stamp_ms = head.uiTimeStamp / 10

        mvsdk.CameraImageProcess(hCamera, pRawData, self.buffer_address, head)
        mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)

        try:
            self.tx.put_nowait((callback_time_s, camera_stamp_ms))
        except queue.Full:
            try:
                self.tx.get_nowait()
            except queue.Empty:
                pass
            finally:
                try:
                    self.tx.put_nowait((callback_time_s, camera_stamp_ms))
                except queue.Full:
                    print(f'Camera Queue Full! Successed Count: {self.success_count}')
                    self.success_count = -1

        self.success_count += 1
