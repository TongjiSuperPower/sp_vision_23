import time
import queue
import logging
from collections import deque
from multiprocessing import Process, Queue
from modules.io.communication import Communicator, Command, TX_FLAG_EMPTY, TX_FLAG_FIRE
from modules.io.context_manager import ContextManager
from modules.tools import clear_queue


def transmit(port: str, rx_queue: Queue, quit_queue: Queue) -> None:
    logging.info('Transmit process started.')

    scheduled_command: Command = None
    scheduled_time_s: float = None
    with Communicator(port, use_rx=False) as communicator:
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
                    command, fire_time_s = rx_queue.get_nowait()
                    communicator.send(*command)

                    x, y, z, _ = command
                    scheduled_command = (x, y, z, TX_FLAG_FIRE)

                    if scheduled_time_s is None:
                        scheduled_time_s = fire_time_s

                except queue.Empty:
                    pass

                # 发送定时命令
                if scheduled_time_s is not None:
                    current_time_s = time.time()
                    dt_s = scheduled_time_s - current_time_s
                    if dt_s < 0:
                        print(f'Scheduled command expired over {-dt_s}s.')
                        scheduled_time_s = None
                    elif dt_s > 1:
                        print(f'Scheduled command is so far over {dt_s}s.')
                        scheduled_time_s = None
                    elif dt_s < 1e-3:
                        print(f'Scheduled command sent.')
                        communicator.send(*scheduled_command)
                        scheduled_time_s = None

            except OSError:
                logging.warning('TxCommunicator lost.')
                communicator.reopen()

    clear_queue(rx_queue)
    clear_queue(quit_queue)

    logging.info('Transmit process ended.')


class ParallelTxCommunicator(ContextManager):
    def __init__(self, port: str) -> None:
        self._tx_queue = Queue(maxsize=1)
        self._quit_queue = Queue()
        self._process = Process(target=transmit, args=(port, self._tx_queue, self._quit_queue))

        self._process.start()

    def _close(self) -> None:
        '''注意阻塞'''
        self._quit_queue.put(True)
        self._process.join()
        logging.info('ParallelTxCommunicator closed.')

    def send(self, x_in_imu_mm: float, y_in_imu_mm: float, z_in_imu_mm: float, flag: int = TX_FLAG_EMPTY, fire_time_s: float | None = None) -> None:
        command = (x_in_imu_mm, y_in_imu_mm, z_in_imu_mm, flag)
        try:
            self._tx_queue.put_nowait((command, fire_time_s))
        except queue.Full:
            logging.debug(f'ParallelCommunicator tx_queue full!')
