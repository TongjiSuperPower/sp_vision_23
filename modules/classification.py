import cv2
import tensorflow as tf


class Classifier:
    def __init__(self) -> None:
        self.model = tf.keras.models.load_model('assets/model.h5')

    def classify(self, img: cv2.Mat) -> int:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (32, 32))
        img = img / 255

        tensor = tf.convert_to_tensor(img, dtype=tf.float32)
        tensor = tf.expand_dims(tensor, 0)

        result = self.model(tensor).numpy()

        return result.argmax()