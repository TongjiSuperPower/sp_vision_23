import cv2
import time
import queue
import datetime
from multiprocessing import Process, Queue
from modules.tools import clear_queue

def recording(path: str, name: str, fps: int, informations: Queue, quit_queue: Queue):
    try:
        img, state = informations.get()
        h, w, _ = img.shape
        
        video_path = f'{path}/{name}.avi'
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
        video_writer.write(img)

        state_path = f'{path}/{name}.txt'
        state_writer = open(state_path, 'w')
        state_writer.write(f'{state}\n')

        while True:
            # 判断是否退出
            try:
                quit = quit_queue.get_nowait()
                if quit:
                    break
            except queue.Empty:
                pass

            try:
                img, state = informations.get_nowait()
                video_writer.write(img)
                state_writer.write(f'{state}\n')
            except queue.Empty:
                pass
    
    except KeyboardInterrupt:
        pass

    finally:
        video_writer.release()
        print(f'Video is saved at {video_path}')
        state_writer.close()
        print(f'State is saved at {state_path}')


class Recorder:
    def __init__(self, fps: int = 30, path: str = 'assets/recordings') -> None:
        self.fps = fps
        self.path = path
        self.name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.infomations = Queue()
        self.quit = Queue()
        self.recording = Process(
            target=recording,
            args=(self.path, self.name, self.fps, self.infomations, self.quit)
        )

        self.recording.start()
        self.last_put_time = 0

    def record(self, img: cv2.Mat, state: list) -> None:
        current_time = time.time()
        if 1 / (current_time - self.last_put_time) > self.fps:
            return
        
        self.infomations.put((img, state))
        self.last_put_time = current_time

    def __enter__(self) -> 'Recorder':
        return self

    def __exit__(self, exc_type, exc_value, exc_tb) -> bool:
        # 按ctrl+c所引发的KeyboardInterrupt，判断为手动退出，不打印报错信息
        ignore_error = (exc_type == KeyboardInterrupt)

        self.quit.put(True)
        self.recording.join()

        clear_queue(self.quit)
        clear_queue(self.infomations)

        print('Recorder closed.')

        return ignore_error