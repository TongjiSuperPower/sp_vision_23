from modules.io.mindvision import Camera
from remote_visualizer import Visualizer

if __name__ == '__main__':
    with Camera(4) as camera, Visualizer() as visualizer:
        while True:
            success, img = camera.read()
            if not success:
                continue

            visualizer.show(img)
