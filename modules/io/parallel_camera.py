import cv2
import time
import queue
import ctypes
import logging
import numpy as np
from multiprocessing import Process, Queue, RawArray
from modules.io.mindvision import Camera
from modules.io.context_manager import ContextManager
from modules.tools import clear_queue


def capture(exposure_ms: float, buffers: list[RawArray], tx_queue: Queue, quit_queue: Queue) -> None:
    logging.info('Capture started.')

    with Camera(exposure_ms) as camera:
        successed_count = 0
        last_buffer_index = -1

        while True:
            time.sleep(1e-4)

            # 判断是否退出
            try:
                quit = quit_queue.get_nowait()
                if quit:
                    break
            except queue.Empty:
                pass

            buffer_index = last_buffer_index + 1
            if buffer_index == len(buffers):
                buffer_index = 0

            buffer_address = ctypes.addressof(buffers[buffer_index])
            last_buffer_index = buffer_index

            success, _ = camera.read(buffer_address)
            if not success:
                logging.warning('Camera lost.')
                camera.reopen()
                continue

            try:
                tx_queue.put_nowait((camera.read_time_s, buffer_index))
            except queue.Full:
                logging.debug(f'Capture tx_queue full! Successed count: {successed_count}')
                successed_count = -1
            successed_count += 1

    clear_queue(tx_queue)
    clear_queue(quit_queue)

    logging.info('Capture ended.')


class ParallelCamera(ContextManager):
    def __init__(self, exposure_ms: float) -> None:
        self._height = 1024
        self._width = 1280
        self._channel = 3
        self._buffer_num = 3

        img_size = self._height * self._width * self._channel
        self._buffers = [RawArray(ctypes.c_uint8, img_size) for _ in range(self._buffer_num)]
        self._quit_queue = Queue()
        self._rx_queue = Queue(maxsize=1)
        self._process = Process(target=capture, args=(exposure_ms, self._buffers, self._rx_queue, self._quit_queue))

        self._process.start()

        self.img: cv2.Mat = None
        self.read_time_s: float = None

    def _close(self) -> None:
        '''注意阻塞'''
        self._quit_queue.put(True)
        self._process.join()
        logging.info('ParallelCamera closed.')

    def update(self) -> None:
        '''注意阻塞'''
        self.read_time_s, buffer_index = self._rx_queue.get()
        img = np.frombuffer(self._buffers[buffer_index], dtype=np.uint8)
        self.img = img.reshape((self._height, self._width, self._channel))
