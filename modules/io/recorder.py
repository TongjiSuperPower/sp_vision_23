import os
import cv2
import time
import queue
import logging
import datetime
import numpy as np
from multiprocessing import Process, Queue, shared_memory
from modules.tools import clear_queue
from modules.io.context_manager import ContextManager


H, W = 1024, 1280
FPS = 60
DIR = 'recordings'
BUFFER_NUM = 2


def record(buffer_names: tuple[str], index_with_status_queue: Queue, quit_queue: Queue) -> None:
    logging.info('Record started.')

    imgs: list[cv2.Mat] = []
    buffers: list[shared_memory.SharedMemory] = []
    for name in buffer_names:
        buffer = shared_memory.SharedMemory(name=name)
        img = np.ndarray((H, W, 3), np.uint8, buffer.buf)
        buffers.append(buffer)
        imgs.append(img)

    if not os.path.isdir(DIR):
        os.mkdir(DIR)

    filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    video_path = f'{DIR}/{filename}.avi'
    status_path = f'{DIR}/{filename}.txt'

    video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'MJPG'), FPS, (W, H))
    status_writer = open(status_path, 'w')

    try:
        while True:
            # 判断是否退出
            try:
                quit = quit_queue.get_nowait()
                if quit:
                    break
            except queue.Empty:
                pass

            try:
                index, status = index_with_status_queue.get()
                img = imgs[index]
                video_writer.write(img)
                status_writer.write(f'{status}\n')
            except queue.Empty:
                pass

    except KeyboardInterrupt:
        pass

    finally:
        video_writer.release()
        logging.info(f'Video is saved at {video_path}')
        status_writer.close()
        logging.info(f'State is saved at {status_path}')

    for buffer in buffers:
        buffer.close()

    clear_queue(index_with_status_queue)
    clear_queue(quit_queue)

    logging.info('Record ended.')


class Recorder(ContextManager):
    def __init__(self) -> None:
        buffer_names: list[str] = []
        self._imgs: list[cv2.Mat] = []
        self._buffers: list[shared_memory.SharedMemory] = []
        for _ in range(BUFFER_NUM):
            buffer = shared_memory.SharedMemory(create=True, size=H*W*3)
            img = np.ndarray((H, W, 3), np.uint8, buffer.buf)
            self._buffers.append(buffer)
            buffer_names.append(buffer.name)
            self._imgs.append(img)

        self._index_status_queue = Queue()
        self._quit_queue = Queue()
        self._process = Process(
            target=record,
            args=(buffer_names, self._index_status_queue, self._quit_queue)
        )

        self._process.start()
        self._last_put_time = 0
        self._last_index = -1

    def _close(self) -> None:
        '''注意阻塞'''
        self._quit_queue.put(True)
        self._process.join()

        for buffer in self._buffers:
            buffer.close()
            buffer.unlink()

        logging.info('Recorder closed.')

    def record(self, img: cv2.Mat, status: list) -> None:
        current_time = time.time()
        if 1 / (current_time - self._last_put_time) > FPS:
            return
        
        index = self._last_index + 1
        if index == len(self._buffers):
            index = 0

        self._imgs[index][:] = img[:]
        self._index_status_queue.put((index, status))

        self._last_put_time = current_time
        self._last_index = index
