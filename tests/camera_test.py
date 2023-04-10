from remote_visualizer import Visualizer
from modules.io.mindvision import Camera

with Camera(3, True) as camera, Visualizer(enable=False) as visualizer:
    last = 0
    
    while True:
        success, img = camera.read()
        if not success:
            continue

        stamp = camera.get_stamp_ms()
        print(f'{stamp-last}ms')
        last = stamp

        visualizer.show(img)
        