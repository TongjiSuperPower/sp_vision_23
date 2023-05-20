import datetime
import parent_folder
from modules.io.communication import Communicator, yaw_pitch_to_xyz, TX_FLAG_FIRE, TX_FLAG_EMPTY

with Communicator('/dev/ttyUSB0') as communicator:
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
            fire = input('是否开火[y/n]:')
            if fire == 'y' or fire == 'n':
                break
            else:
                print('请重新输入')

        flag = TX_FLAG_FIRE if fire == 'y' else TX_FLAG_EMPTY
        x, y, z = yaw_pitch_to_xyz(yaw, pitch)
        communicator.send(x, y, z, flag, debug=True)

    # 测试接收
    last_read_time_s = 0
    while test == 'rx':
        communicator.read(debug=True)
        print(f'{datetime.datetime.now()} {(communicator.read_time_s - last_read_time_s)*1e3:.2f}ms')
        last_read_time_s = communicator.read_time_s
