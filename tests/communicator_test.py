from modules.communication import Communicator

communicator = Communicator('/dev/tty.usbserial-1110')

while True:
    print(communicator.receive())