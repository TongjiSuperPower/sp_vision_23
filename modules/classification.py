import cv2
import numpy as np

class_names = ('big_one', 'big_three', 'big_four', 'big_five', 'big_base',
               'small_two', 'small_three', 'small_four', 'small_five',
               'small_base', 'small_santry', 'small_outpost',
               'no_pattern')


class Classifier:
    def __init__(self) -> None:
        self.net = cv2.dnn.readNetFromONNX('assets/model.onnx')

    def classify(self, pattern_img: cv2.Mat) -> tuple[float, str]:
        pattern_img = cv2.resize(pattern_img, (50, 50))
        pattern_img = pattern_img.astype(np.float32)
        pattern_img = pattern_img / 255
        pattern_img = pattern_img.reshape((1, 1, 50, 50))

        self.net.setInput(pattern_img)
        out = self.net.forward()
        out = np.exp(out) / np.sum(np.exp(out))  # softmax
        class_id = np.argmax(out)
        confidence = out[0][class_id]

        return confidence, class_names[class_id]
