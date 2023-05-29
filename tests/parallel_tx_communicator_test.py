import time
import datetime
import parent_folder
from modules.io.communication import yaw_pitch_to_xyz
from modules.io.parallel_tx_communicator import ParallelTxCommunicator

if __name__ == '__main__':
    with ParallelTxCommunicator('/dev/ttyUSB0') as communicator:
        while True:
            while True:
                try:
                    yaw, pitch = map(float, input('请输入yaw空格pitch单位degree:').split())
                    break
                except ValueError:
                    print('请重新输入')

            while True:
                fire = input('是否1秒后开火[y/n]:')
                if fire == 'y' or fire == 'n':
                    break
                else:
                    print('请重新输入')
            fire_time_s = time.time() + 1 if fire == 'y' else None

            x, y, z = yaw_pitch_to_xyz(yaw, pitch)
            communicator.send(x, y, z, fire_time_s=fire_time_s)
