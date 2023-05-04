import parent_folder
from modules.io.mindvision import Camera
from remote_visualizer import Visualizer

if __name__ == '__main__':
    with Camera(4) as camera, Visualizer(enable=True) as visualizer:
        last_stamp_ms = 0
        while True:
            success, img = camera.read()
            if not success:
                continue

            stamp_ms = camera.get_stamp_ms()
            print(f'{stamp_ms - last_stamp_ms}ms')
            last_stamp_ms = stamp_ms

            visualizer.show(img)
