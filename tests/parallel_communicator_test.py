import time
import datetime
import parent_folder
from remote_visualizer import Visualizer
from modules.io.communication import yaw_pitch_to_xyz
from modules.io.parallel_communicator import ParallelCommunicator

if __name__ == '__main__':
    with ParallelCommunicator('/dev/ttyUSB0') as communicator:
        test: str = None

        while True:
            test = input('测试发送/接收?输入[tx/rx]\n')
            if test == 'rx' or test == 'tx':
                break
            else:
                print('请重新输入')

        # 测试发送
        while test == 'tx':
            while True:
                try:
                    yaw, pitch = map(float, input('请输入yaw空格pitch单位degree:').split())
                    break
                except ValueError:
                    print('请重新输入')

            while True:
                fire = input('是否5秒后开火[y/n]:')
                if fire == 'y' or fire == 'n':
                    break
                else:
                    print('请重新输入')
            fire_time_s = time.time() + 5 if fire == 'y' else None

            x, y, z = yaw_pitch_to_xyz(yaw, pitch)
            communicator.send(fire_time_s, x, y, z)

        # 测试接收
        last_read_time_s = 0
        while True:
            communicator.update()
            print(f'{datetime.datetime.now()} {(communicator.latest_read_time_s - last_read_time_s)*1e3:.2f}ms {communicator.latest_status}')
            last_read_time_s = communicator.latest_read_time_s
