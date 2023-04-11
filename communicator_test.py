from modules.io.communication import Communicator

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
        try:
            yaw, pitch = map(float, input('请输入yaw空格pitch:').split())
            communicator.send_yaw_pitch(yaw, pitch)
        except KeyboardInterrupt:
            break
        except:
            print('请重新输入')

    # 测试接收
    while test == 'rx':
        communicator.receive(debug=True)
