import cv2
import modules.mvsdk as mvsdk
import numpy as np


class Camera:
    def __init__(self, exposureMs: float = None) -> None:
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
        mvsdk.CameraSetTriggerMode(self.camera, mvsdk.FALSE)  # 连续采集模式
        mvsdk.CameraSetFrameSpeed(self.camera, mvsdk.FRAME_SPEED_HIGH)  # 高帧率模式
        if exposureMs != None:
            mvsdk.CameraSetAeState(self.camera, mvsdk.FALSE)  # 手动曝光
            mvsdk.CameraSetExposureTime(self.camera, exposureMs * 1000)  # 曝光时间ms

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

    def getTimeStampUs(self) -> int:
        return mvsdk.CameraGetFrameTimeStamp(self.camera)

    def release(self) -> None:
        # 关闭相机
        mvsdk.CameraUnInit(self.camera)
        # 释放帧缓存
        mvsdk.CameraAlignFree(self.buffer)
