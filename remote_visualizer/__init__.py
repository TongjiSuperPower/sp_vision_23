import cv2
import time
import queue
from multiprocessing import Process, Queue


def clear_queue(q: Queue) -> None:
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        return


def get_local_ip():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('192.255.255.255', 1))
        my_ip = s.getsockname()[0]
    except:
        my_ip = '127.0.0.1'
    finally:
        s.close()
    return my_ip


def visualizing(port: int, show_queue: Queue, plot_queue: Queue):
    import json
    import logging
    from flask import Flask, Response, render_template, make_response

    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    app = Flask(__name__)

    @app.route('/data')
    def data():
        data = plot_queue.get()
        response = make_response(json.dumps(data))
        response.content_type = 'application/json'
        return response

    @app.route('/plot')
    def plot():
        return render_template('plot.html')

    @app.route('/video_feed')
    def video_feed():
        def next_frame():
            while True:
                img = show_queue.get()
                _, buffer = cv2.imencode('.jpg', img)
                img = buffer.tobytes()
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+img+b'\r\n'

        return Response(next_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/')
    def index():
        return render_template('index.html')

    app.run(host="0.0.0.0", port=port, debug=False)


class Visualizer:
    def __init__(self, port: int = 60000, enable: bool = True, fps: int = 30) -> None:
        self.enable = enable
        if not self.enable:
            return

        self._show_queue = Queue(maxsize=1)
        self._plot_queue = Queue(maxsize=1)
        self._plot_buffer = []

        self.visualizing = Process(
            target=visualizing,
            args=(port, self._show_queue, self._plot_queue)
        )

        self.visualizing.start()
        host_ip = get_local_ip()
        print(f'\n * Visualizer will be running on http://{host_ip}:{port}')

        self.fps = fps
        self.last_put_time = 0

    def show(self, img: cv2.Mat) -> None:
        if not self.enable:
            return

        current_time = time.time()
        if 1 / (current_time - self.last_put_time) > self.fps:
            return

        try:
            self._show_queue.put_nowait(img)
            self.last_put_time = current_time
        except queue.Full:
            pass

    def plot(self, values=(), names=()) -> None:
        if not self.enable:
            return

        self._plot_buffer.append({'values': values, 'names': names})
        try:
            self._plot_queue.put_nowait(self._plot_buffer)
            self._plot_buffer = []
        except queue.Full:
            pass

    def __enter__(self) -> 'Visualizer':
        return self

    def __exit__(self, exc_type, exc_value, exc_tb) -> bool:
        # 按ctrl+c所引发的KeyboardInterrupt，判断为手动退出，不打印报错信息
        ignore_error = (exc_type == KeyboardInterrupt)

        if not self.enable:
            return ignore_error

        self.visualizing.terminate()
        self.visualizing.join()

        clear_queue(self._plot_queue)
        clear_queue(self._show_queue)

        print('Visualizer closed.')

        return ignore_error
