import datetime
import parent_folder
from modules.io.communication import Communicator

with Communicator('/dev/tty.usbserial-140') as communicator:
    test: str = None

    while True:
        test = input('测试发送/接收?输入[tx/rx]\n')
        if test == 'rx' or test == 'tx':
            break
        else:
            print('请重新输入')

    # 测试发送
    while test == 'tx':
        try:
            yaw, pitch = map(float, input('请输入yaw空格pitch:').split())
            communicator.send_yaw_pitch(yaw, pitch)
        except KeyboardInterrupt:
            break
        except:
            print('请重新输入')

    # 测试接收
    last_read_time_s = 0
    while test == 'rx':
        communicator.read(debug=True)
        print(f'{datetime.datetime.now()} {(communicator.read_time_s - last_read_time_s)*1e3:.2f}ms')
        last_read_time_s = communicator.read_time_s
