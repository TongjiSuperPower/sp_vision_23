import time
import queue
import logging
from collections import deque
from multiprocessing import Process, Queue
from modules.io.communication import Communicator, Status, TX_FLAG_FIRE
from modules.io.context_manager import ContextManager
from modules.tools import clear_queue


def communicate(port: str, tx_queue: Queue, rx_queue: Queue, quit_queue: Queue) -> None:
    logging.info('Communicate started.')

    buffer = []
    scheduled_command: tuple = None
    scheduled_time_s: float = None
    with Communicator(port) as communicator:
        while True:
            time.sleep(1e-4)

            try:
                # 判断是否退出
                try:
                    quit = quit_queue.get_nowait()
                    if quit:
                        break
                except queue.Empty:
                    pass

                # 发送命令
                try:
                    xyz, fire_time_s = rx_queue.get_nowait()
                    if type(fire_time_s) == str and fire_time_s == 'now':
                        communicator.send(*xyz, flag=TX_FLAG_FIRE)
                    else:
                        communicator.send(*xyz)

                    x, y, z = xyz
                    scheduled_command = (x, y, z, TX_FLAG_FIRE)                        

                    if scheduled_time_s is None and type(fire_time_s) != str:
                        scheduled_time_s = fire_time_s

                except queue.Empty:
                    pass

                current_time_s = time.time()
                if scheduled_time_s is None:
                    pass
                elif scheduled_time_s - current_time_s < 0:
                    scheduled_time_s = None
                elif scheduled_time_s - current_time_s > 1:
                    scheduled_time_s = None
                elif scheduled_time_s - current_time_s < 1e-3:
                    communicator.send(*scheduled_command)
                    scheduled_time_s = None

                # 接收机器人状态
                success, status = communicator.read_no_wait(debug=False)
                if not success:
                    continue

                buffer.append((communicator.read_time_s, status))

                try:
                    tx_queue.put_nowait(buffer)
                    buffer = []
                except queue.Full:
                    pass

            except OSError:
                logging.warning('Communicator lost.')
                communicator.reopen()

    clear_queue(tx_queue)
    clear_queue(rx_queue)
    clear_queue(quit_queue)

    logging.info('Communicate ended.')


class ParallelCommunicator(ContextManager):
    def __init__(self, port: str) -> None:
        self._rx_queue = Queue(maxsize=1)
        self._tx_queue = Queue(maxsize=1)
        self._quit_queue = Queue()
        self._process = Process(target=communicate, args=(port, self._rx_queue, self._tx_queue, self._quit_queue))

        self._process.start()

        self.history: deque[tuple[float, Status]] = deque(maxlen=1000)
        self.latest_read_time_s: float = None
        self.latest_status: Status = None

    def _close(self) -> None:
        '''注意阻塞'''
        self._quit_queue.put(True)
        self._process.join()
        logging.info('ParallelCommunicator closed.')

    def update(self) -> None:
        '''注意阻塞'''
        buffer = self._rx_queue.get()
        self.history.extend(buffer)
        self.latest_read_time_s, self.latest_status = self.history[-1]

    def send(self, x_in_imu_mm: float, y_in_imu_mm: float, z_in_imu_mm: float, fire_time_s: float | str | None) -> None:
        xyz = (x_in_imu_mm, y_in_imu_mm, z_in_imu_mm)
        try:
            self._tx_queue.put_nowait((xyz, fire_time_s))
        except queue.Full:
            logging.debug(f'ParallelCommunicator tx_queue full!')
