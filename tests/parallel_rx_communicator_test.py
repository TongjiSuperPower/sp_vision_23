import datetime
import parent_folder
from modules.io.parallel_rx_communicator import ParallelRxCommunicator

if __name__ == '__main__':
    with ParallelRxCommunicator('/dev/ttyUSB0') as communicator:
        last_read_time_s = 0
        while True:
            communicator.update()
            print(f'{datetime.datetime.now()} {(communicator.latest_read_time_s - last_read_time_s)*1e3:.2f}ms {communicator.latest_status}')
            last_read_time_s = communicator.latest_read_time_s
