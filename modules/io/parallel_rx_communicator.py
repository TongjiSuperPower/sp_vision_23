import time
import queue
import logging
from collections import deque
from multiprocessing import Process, Queue
from modules.io.communication import Communicator, Status
from modules.io.context_manager import ContextManager
from modules.tools import clear_queue


def receive(port: str, tx_queue: Queue, quit_queue: Queue) -> None:
    logging.info('Receive process started.')

    buffer = []
    with Communicator(port, ues_tx=False) as communicator:
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
                logging.warning('RxCommunicator lost.')
                communicator.reopen()

    clear_queue(tx_queue)
    clear_queue(quit_queue)

    logging.info('Receive process ended.')


class ParallelRxCommunicator(ContextManager):
    def __init__(self, port: str) -> None:
        self._rx_queue = Queue(maxsize=1)
        self._quit_queue = Queue()
        self._process = Process(target=receive, args=(port, self._rx_queue, self._quit_queue))

        self._process.start()

        self.history: deque[tuple[float, Status]] = deque(maxlen=1000)
        self.latest_read_time_s: float = None
        self.latest_status: Status = None

    def _close(self) -> None:
        '''注意阻塞'''
        self._quit_queue.put(True)
        self._process.join()
        logging.info('ParallelRxCommunicator closed.')

    def update(self) -> None:
        '''注意阻塞'''
        buffer = self._rx_queue.get()
        self.history.extend(buffer)
        self.latest_read_time_s, self.latest_status = self.history[-1]
