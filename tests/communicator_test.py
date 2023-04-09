from modules.communication import Communicator

with Communicator('/dev/ttyUSB0') as communicator:

    # 测试发送
    while True:
        try:
            yaw, pitch = map(float, input().split())
            communicator.send_yaw_pitch(yaw, pitch)
        except KeyboardInterrupt:
            break
        except:
            pass

    # 测试接收
    # while True:
    #     print(communicator.receive())
