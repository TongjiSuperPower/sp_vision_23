import queue
from collections import deque
from multiprocessing import Process, Queue
from modules.io.communication import Communicator, Status
from modules.tools import clear_queue, ContextManager


def communicate(port: str, tx_queue: Queue, rx_queue: Queue, quit_queue: Queue) -> None:
    print('Communicate started.')

    buffer = []
    with Communicator(port) as communicator:
        while True:
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
                    command = rx_queue.get_nowait()
                    communicator.send(*command)
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
                print('Communicator lost.')
                communicator.reopen()

    clear_queue(tx_queue)
    clear_queue(rx_queue)
    clear_queue(quit_queue)

    print('Communicate ended.')


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
        print('ParallelCommunicator closed.')

    def update(self) -> None:
        '''注意阻塞'''
        buffer = self._rx_queue.get()
        self.history.extend(buffer)
        self.latest_read_time_s, self.latest_status = self.history[-1]

    def send(
        self,
        x_in_imu: float = 0, y_in_imu: float = 0, z_in_imu: float = 0,
        vx_in_imu: float = 0, vy_in_imu: float = 0, vz_in_imu: float = 0,
        stamp: int = 0, flag: int = 0,
    ) -> None:
        command = (x_in_imu, y_in_imu, z_in_imu, vx_in_imu, vy_in_imu, vz_in_imu, stamp, flag)
        try:
            self._tx_queue.put_nowait(command)
        except queue.Full:
            print(f'ParallelCommunicator tx_queue full!')
