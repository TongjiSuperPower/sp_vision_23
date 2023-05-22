import os
import cv2
import time
import queue
import logging
import datetime
from multiprocessing import Process, Queue
from modules.tools import clear_queue
from modules.io.context_manager import ContextManager


def record(dir: str, fps: int, img_status_queue: Queue, quit_queue: Queue) -> None:
    logging.info('Record started.')

    try:
        filename = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        img, status = img_status_queue.get()
        h, w, _ = img.shape

        video_path = f'{dir}/{filename}.avi'
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
        video_writer.write(img)

        status_path = f'{dir}/{filename}.txt'
        status_writer = open(status_path, 'w')
        status_writer.write(f'{status}\n')

        while True:
            # 判断是否退出
            try:
                quit = quit_queue.get_nowait()
                if quit:
                    break
            except queue.Empty:
                pass

            try:
                img, status = img_status_queue.get_nowait()
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

    clear_queue(img_status_queue)
    clear_queue(quit_queue)

    logging.info('Record ended.')


class Recorder(ContextManager):
    def __init__(self, fps: int = 60, dir: str = 'recordings') -> None:
        self._fps = fps
        self._img_status_queue = Queue()
        self._quit_queue = Queue()
        self._process = Process(
            target=record,
            args=(dir, self._fps, self._img_status_queue, self._quit_queue)
        )

        self._process.start()
        self.last_put_time = 0

    def _close(self) -> None:
        '''注意阻塞'''
        self._quit_queue.put(True)
        self._process.join()
        logging.info('Recorder closed.')

    def record(self, img: cv2.Mat, status: list) -> None:
        current_time = time.time()
        if 1 / (current_time - self.last_put_time) > self._fps:
            return

        self._img_status_queue.put((img, status))
        self.last_put_time = current_time
