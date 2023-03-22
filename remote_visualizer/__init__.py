import cv2
import queue
from multiprocessing import Process, Queue


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


def visualizing(port: int, frame_queue: Queue, plot_queue: Queue):
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
                frame = frame_queue.get()
                _, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+frame+b'\r\n'

        return Response(next_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/')
    def index():
        return render_template('index.html')

    app.run(host="0.0.0.0", port=port, debug=False)


class Visualizer:
    def __init__(self, port: int = 60000) -> None:
        self._frame_queue = Queue(maxsize=1)
        self._plot_queue = Queue(maxsize=1)
        self._plot_buffer = []

        self._process = Process(
            target=visualizing,
            args=(port, self._frame_queue, self._plot_queue)
        )

        self._process.start()
        host_ip = get_local_ip()
        print(f'\n * Remote Visualizer will be running on http://{host_ip}:{port}')

    def show(self, img: cv2.Mat) -> None:
        try:
            self._frame_queue.put_nowait(img)
        except queue.Full:
            pass

    def plot(self, values=(), names=()) -> None:
        self._plot_buffer.append({'values': values, 'names': names})
        try:
            self._plot_queue.put_nowait(self._plot_buffer)
            self._plot_buffer = []
        except queue.Full:
            pass
