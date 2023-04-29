import math
import serial
import struct


frame_head = 0xf1
frame_tail = 0xf2
frame_tx_len = 17
frame_rx_len = 11


crc8Table = [
    0x00, 0x5e, 0xbc, 0xe2, 0x61, 0x3f, 0xdd, 0x83, 0xc2, 0x9c, 0x7e, 0x20, 0xa3, 0xfd, 0x1f, 0x41,
    0x9d, 0xc3, 0x21, 0x7f, 0xfc, 0xa2, 0x40, 0x1e, 0x5f, 0x01, 0xe3, 0xbd, 0x3e, 0x60, 0x82, 0xdc,
    0x23, 0x7d, 0x9f, 0xc1, 0x42, 0x1c, 0xfe, 0xa0, 0xe1, 0xbf, 0x5d, 0x03, 0x80, 0xde, 0x3c, 0x62,
    0xbe, 0xe0, 0x02, 0x5c, 0xdf, 0x81, 0x63, 0x3d, 0x7c, 0x22, 0xc0, 0x9e, 0x1d, 0x43, 0xa1, 0xff,
    0x46, 0x18, 0xfa, 0xa4, 0x27, 0x79, 0x9b, 0xc5, 0x84, 0xda, 0x38, 0x66, 0xe5, 0xbb, 0x59, 0x07,
    0xdb, 0x85, 0x67, 0x39, 0xba, 0xe4, 0x06, 0x58, 0x19, 0x47, 0xa5, 0xfb, 0x78, 0x26, 0xc4, 0x9a,
    0x65, 0x3b, 0xd9, 0x87, 0x04, 0x5a, 0xb8, 0xe6, 0xa7, 0xf9, 0x1b, 0x45, 0xc6, 0x98, 0x7a, 0x24,
    0xf8, 0xa6, 0x44, 0x1a, 0x99, 0xc7, 0x25, 0x7b, 0x3a, 0x64, 0x86, 0xd8, 0x5b, 0x05, 0xe7, 0xb9,
    0x8c, 0xd2, 0x30, 0x6e, 0xed, 0xb3, 0x51, 0x0f, 0x4e, 0x10, 0xf2, 0xac, 0x2f, 0x71, 0x93, 0xcd,
    0x11, 0x4f, 0xad, 0xf3, 0x70, 0x2e, 0xcc, 0x92, 0xd3, 0x8d, 0x6f, 0x31, 0xb2, 0xec, 0x0e, 0x50,
    0xaf, 0xf1, 0x13, 0x4d, 0xce, 0x90, 0x72, 0x2c, 0x6d, 0x33, 0xd1, 0x8f, 0x0c, 0x52, 0xb0, 0xee,
    0x32, 0x6c, 0x8e, 0xd0, 0x53, 0x0d, 0xef, 0xb1, 0xf0, 0xae, 0x4c, 0x12, 0x91, 0xcf, 0x2d, 0x73,
    0xca, 0x94, 0x76, 0x28, 0xab, 0xf5, 0x17, 0x49, 0x08, 0x56, 0xb4, 0xea, 0x69, 0x37, 0xd5, 0x8b,
    0x57, 0x09, 0xeb, 0xb5, 0x36, 0x68, 0x8a, 0xd4, 0x95, 0xcb, 0x29, 0x77, 0xf4, 0xaa, 0x48, 0x16,
    0xe9, 0xb7, 0x55, 0x0b, 0x88, 0xd6, 0x34, 0x6a, 0x2b, 0x75, 0x97, 0xc9, 0x4a, 0x14, 0xf6, 0xa8,
    0x74, 0x2a, 0xc8, 0x96, 0x15, 0x4b, 0xa9, 0xf7, 0xb6, 0xe8, 0x0a, 0x54, 0xd7, 0x89, 0x6b, 0x35,
]


def calculateCrc8(data: bytes) -> int:
    crc = 0xff
    index = 0
    for i in range(len(data)):
        index = crc ^ data[i]
        crc = crc8Table[index]
    return crc


def pack_frame(
    stamp: int,
    x_in_imu: float, y_in_imu: float, z_in_imu: float,
    vx_in_imu: float, vy_in_imu: float, vz_in_imu: float,
    flag: int
) -> bytes:

    x_in_imu = int(x_in_imu)
    y_in_imu = int(y_in_imu)
    z_in_imu = int(z_in_imu)
    vx_in_imu = int(vx_in_imu * 1e3)
    vy_in_imu = int(vy_in_imu * 1e3)
    vz_in_imu = int(vz_in_imu * 1e3)

    crc_part = struct.pack('=BBhhhhhhB', frame_head, stamp, x_in_imu, y_in_imu, z_in_imu, vx_in_imu, vy_in_imu, vz_in_imu, flag)
    crc = calculateCrc8(crc_part)
    frame = crc_part + struct.pack('BB', crc, frame_tail)
    return frame


def unpack_frame(frame: bytes) -> tuple[bool, None | tuple[int, float, float, float, int]]:
    head, stamp, yaw, pitch, bullet_speed, flag, crc, tail = struct.unpack('=BBhhhBBB', frame)

    if head != frame_head or tail != frame_tail or crc != calculateCrc8(frame[:-2]):
        return False, None

    yaw = yaw / 1e2
    pitch = pitch / 1e2
    bullet_speed = bullet_speed / 1e2

    return True, (stamp, yaw, pitch, bullet_speed, flag)


def yaw_pitch_to_xyz(yaw: float, pitch: float) -> tuple[float, float, float]:
    yaw, pitch = math.radians(yaw), math.radians(pitch)
    y = -math.sin(pitch)
    xz = math.cos(pitch)
    x = xz * math.sin(yaw)
    z = xz * math.cos(yaw)
    return x, y, z


class Communicator:
    def __init__(self, port: str) -> None:
        self.serial = serial.Serial(port, 115200)
        self.serial.reset_input_buffer()
        self.serial.reset_output_buffer()

    def send(
        self,
        x_in_imu: float = 0, y_in_imu: float = 0, z_in_imu: float = 0,
        vx_in_imu: float = 0, vy_in_imu: float = 0, vz_in_imu: float = 0,
        stamp: int = 0, flag: int = 0,
        debug: bool = False
    ) -> None:

        frame = pack_frame(stamp, x_in_imu, y_in_imu, z_in_imu, vx_in_imu, vy_in_imu, vz_in_imu, flag)
        self.serial.write(frame)

        if debug:
            print(f'sent {stamp=} x={x_in_imu} y={y_in_imu} z={z_in_imu} vx={vx_in_imu} vy={vy_in_imu} vz={vz_in_imu} {flag=} {frame.hex()}')

    def send_yaw_pitch(self, yaw: float, pitch: float, debug: bool = False) -> None:
        '''旋转正方向符合对应坐标轴右手螺旋'''
        x, y, z = yaw_pitch_to_xyz(yaw, pitch)
        self.send(x*1e3, y*1e3, z*1e3, debug)

    def receive_no_wait(self, debug: bool = False) -> tuple[bool, None | tuple[int, float, float, float, int]]:
        frame = self.serial.read_all()
        if len(frame) != frame_rx_len:
            if len(frame) != 0:
                print(f'failed to receive {frame.hex()}')
            return False, None

        success, state = unpack_frame(frame)
        if not success:
            print(f'failed to unpack {frame.hex()}')
            return False, None

        if debug:
            stamp, yaw, pitch, bullet_speed, flag = state
            print(f'received {stamp=} yaw={yaw:.2f} pitch={pitch:.2f} bullet_speed={bullet_speed:.2f} {flag=} {frame.hex()}')

        return True, state

    def receive(self, debug: bool = False) -> tuple[int, float, float, float, int]:
        while True:
            success, state = self.receive_no_wait(debug)
            if not success:
                continue

            return state

    def __enter__(self) -> 'Communicator':
        return self

    def __exit__(self, exc_type, exc_value, exc_tb) -> bool:
        # 按ctrl+c所引发的KeyboardInterrupt，判断为手动退出，不打印报错信息
        ignore_error = (exc_type == KeyboardInterrupt)

        self.serial.close()
        print('Communicator closed.')

        return ignore_error